"""Step 4 (E5 + E6): learned and recalibrated confidence scores, out-of-fold.

Protocol (same for both):
  * GroupKFold over ``db_id`` (5 folds): a database is never in train and test.
  * Fit only on **clean** questions of the training databases (label: exec_ok).
  * Score clean *and* perturbed questions of the held-out databases.
  So every score is out-of-fold, and perturbed questions are a genuine shift.

Scores written to results/<run>/scores_augment.jsonl (merged by analyze.py):
  p_gm_platt  E6: Platt scaling of logit(p_gm)              (2 parameters)
  probe_lr    E5: L2 logistic regression on hidden states   (linear probe)
  probe_mlp   E5: one-hidden-layer MLP on hidden states     (the NN component)

Also writes probe_ablation.json: linear-probe AUROC for every captured
(layer, position), held-out clean vs held-out perturbed.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from _cli import parse
from nl2sql_conf.config import processed_dir, run_dir
from nl2sql_conf.metrics import auroc
from nl2sql_conf.utils import get_logger, read_jsonl, set_seed, write_jsonl

log = get_logger("augment")
POSITIONS = {"last_prompt": [0], "sql_mean": [1], "both": [0, 1]}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_features(feat_dir) -> dict[str, np.ndarray]:
    """uid -> (n_layers, 2, hidden) float16. Later shards win on duplicate uids."""
    feats = {}
    for shard in sorted(feat_dir.glob("shard_*.npz")):
        z = np.load(shard)
        for uid, f in zip(z["uids"], z["feats"]):
            feats[str(uid)] = f
    return feats


def select(feats: np.ndarray, layer: int, position: str) -> np.ndarray:
    """(N, n_layers, 2, H) -> (N, H or 2H) for one layer and position choice."""
    return feats[:, layer, POSITIONS[position], :].reshape(len(feats), -1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def fit_platt(x_tr, y_tr, x_te):
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(x_tr[:, None], y_tr)
    return lr.predict_proba(x_te[:, None])[:, 1]


def fit_lr_probe(X_tr, y_tr, X_te, C):
    sc = StandardScaler().fit(X_tr)
    lr = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X_tr), y_tr)
    return lr.predict_proba(sc.transform(X_te))[:, 1]


class MLP(torch.nn.Module):
    def __init__(self, d_in: int, hidden: int, dropout: float):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Dropout(dropout), torch.nn.Linear(d_in, hidden), torch.nn.GELU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp_probe(X_tr, y_tr, g_tr, X_te, cfg, seed) -> np.ndarray:
    """MLP with early stopping on a DB-grouped validation split of the training set."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    tr_idx, va_idx = next(GroupShuffleSplit(1, test_size=cfg.val_fraction,
                                            random_state=seed).split(X_tr, y_tr, g_tr))
    sc = StandardScaler().fit(X_tr[tr_idx])
    t = lambda a: torch.tensor(sc.transform(a), dtype=torch.float32, device=device)
    Xa, Xv, Xt = t(X_tr[tr_idx]), t(X_tr[va_idx]), t(X_te)
    ya = torch.tensor(y_tr[tr_idx], dtype=torch.float32, device=device)
    yv = torch.tensor(y_tr[va_idx], dtype=torch.float32, device=device)

    model = MLP(X_tr.shape[1], cfg.hidden, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best, best_state, bad = float("inf"), None, 0
    for _ in range(cfg.max_epochs):          # full-batch: ~1k training examples
        model.train()
        opt.zero_grad()
        loss_fn(model(Xa), ya).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xv), yv).item()
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: x.detach().clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(Xt)).cpu().numpy()


# --------------------------------------------------------------------------- #
def main() -> None:
    _, cfg = parse(__doc__)
    set_seed(cfg.seed)
    a = cfg.augment
    out = run_dir(cfg)

    rows = read_jsonl(out / "labeled.jsonl")
    pairs = read_jsonl(processed_dir(cfg) / "pairs.jsonl")
    clean_uids = {p["clean_uid"] for p in pairs}
    uids = [r["uid"] for r in rows]
    y = np.array([int(r[a.label]) for r in rows])
    groups = np.array([r["db_id"] for r in rows])
    is_clean = np.array([u in clean_uids for u in uids])
    p_gm = np.array([r["p_gm"] for r in rows], dtype=float)
    logit = np.log(np.clip(p_gm, 1e-6, 1 - 1e-6) / (1 - np.clip(p_gm, 1e-6, 1 - 1e-6)))
    logit = np.nan_to_num(logit, nan=0.0)   # empty SQL -> neutral

    feats = load_features(out / "features") if (out / "features").exists() else {}
    have_feats = bool(feats) and all(u in feats for u in uids)
    if not have_feats:
        log.warning("Features missing for %d/%d examples; probes skipped",
                    sum(u not in feats for u in uids), len(uids))
    else:
        F = np.stack([feats[u] for u in uids])                  # (N, n_layers, 2, H)
        fracs = list(cfg.features.layer_fractions)
        layer = int(np.argmin([abs(f - a.probe.feature_layer) for f in fracs]))
        X_main = select(F, layer, a.probe.feature_position)
        log.info("Probe features: layer fraction %.2f, position %s, dim %d",
                 fracs[layer], a.probe.feature_position, X_main.shape[1])

    scores = {k: np.full(len(rows), np.nan) for k in ("p_gm_platt", "probe_lr", "probe_mlp")}
    ablation_scores = {}
    folds = GroupKFold(n_splits=a.n_folds).split(np.zeros(len(rows)), y, groups)
    for k, (tr, te) in enumerate(folds):
        tr = tr[is_clean[tr]]                                   # train on clean only
        if len(set(y[tr])) < 2:
            log.warning("Fold %d has one class in training; skipped", k)
            continue
        scores["p_gm_platt"][te] = fit_platt(logit[tr], y[tr], logit[te])
        if have_feats:
            scores["probe_lr"][te] = fit_lr_probe(X_main[tr], y[tr], X_main[te], a.probe.lr_C)
            scores["probe_mlp"][te] = fit_mlp_probe(X_main[tr], y[tr], groups[tr], X_main[te],
                                                    a.probe.mlp, cfg.seed + k)
            for li, frac in enumerate(cfg.features.layer_fractions):
                for pos in ("last_prompt", "sql_mean"):
                    X = select(F, li, pos)
                    key = f"layer{frac}_{pos}"
                    ablation_scores.setdefault(key, np.full(len(rows), np.nan))
                    ablation_scores[key][te] = fit_lr_probe(X[tr], y[tr], X[te], a.probe.lr_C)
        log.info("fold %d: train %d clean, test %d", k, len(tr), len(te))

    write_jsonl(out / "scores_augment.jsonl", [
        {"uid": u, **{k: (None if np.isnan(v[i]) else float(v[i])) for k, v in scores.items()}}
        for i, u in enumerate(uids)])

    if ablation_scores:
        abl = {k: {"auroc_clean": auroc(v[is_clean], y[is_clean]),
                   "auroc_perturbed": auroc(v[~is_clean], y[~is_clean])}
               for k, v in ablation_scores.items()}
        (out / "probe_ablation.json").write_text(json.dumps(abl, indent=2))
        log.info("Probe ablation (AUROC clean / perturbed, label %s): %s", a.label,
                 {k: (round(v["auroc_clean"], 3), round(v["auroc_perturbed"], 3))
                  for k, v in abl.items()})
    for k, v in scores.items():
        log.info("%s out-of-fold AUROC: clean %.3f, perturbed %.3f", k,
                 auroc(v[is_clean], y[is_clean]), auroc(v[~is_clean], y[~is_clean]))


if __name__ == "__main__":
    main()
