"""Dr.Spider loading, integrity checks, and paired sampling.

Vocabulary used throughout the code base:

* **pair**     one (clean question, perturbed question) twin sharing a gold SQL.
* **example**  one question to run through the model. A clean question that
               appears in several sets (or pairs) is a single example, so the
               model is never asked the identical prompt twice.
* **uid**      content hash of (db_id, question); identical inputs share a uid.
"""

from __future__ import annotations

import hashlib
import json
import random
import tarfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from omegaconf import DictConfig

from .utils import get_logger, repo_path

log = get_logger(__name__)


@dataclass(frozen=True)
class Example:
    uid: str
    db_id: str
    question: str
    gold_sql: str


@dataclass(frozen=True)
class Pair:
    pair_id: str          # "<set>:<index in the set's question files>"
    set_name: str
    spider_qid: int       # index of the clean question in Spider-dev
    db_id: str
    clean_uid: str
    perturbed_uid: str


def make_uid(db_id: str, question: str) -> str:
    """Stable 12-hex-char id; identical (db, question) inputs collide on purpose."""
    return hashlib.sha1(f"{db_id}\x1f{question}".encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Download and extraction
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_raw_data(cfg: DictConfig) -> Path:
    """Download + checksum + extract Dr.Spider once. Returns the extracted root."""
    raw_dir = repo_path(cfg.paths.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / cfg.data.archive_name
    root = raw_dir / cfg.data.extract_dir

    if not archive.exists():
        log.info("Downloading %s", cfg.data.source_url)
        urllib.request.urlretrieve(cfg.data.source_url, archive)

    digest = _sha256(archive)
    if digest != cfg.data.sha256:
        raise RuntimeError(f"Checksum mismatch for {archive}: {digest}")

    if not root.exists():
        log.info("Extracting %s -> %s", archive.name, root)
        root.mkdir(parents=True)
        with tarfile.open(archive) as tar:
            tar.extractall(root)
    return root


def ensure_test_suite(cfg: DictConfig, db_ids: set[str]) -> Path:
    """Download the test-suite zip once and extract only the needed DB folders.

    Returns the directory holding ``<db_id>/*.sqlite`` variants.
    """
    ts = cfg.data.test_suite
    raw_dir = repo_path(cfg.paths.raw_dir)
    archive = raw_dir / ts.archive_name
    root = raw_dir / ts.extract_dir / "database"
    if not archive.exists():
        log.info("Downloading test-suite databases (1.27 GB) ...")
        tmp = archive.with_suffix(".part")
        req = urllib.request.Request(ts.url, headers={"User-Agent": "nl2sql-conf/0.2"})
        with urllib.request.urlopen(req) as r, tmp.open("wb") as f:
            for chunk in iter(lambda: r.read(1 << 22), b""):
                f.write(chunk)
        tmp.rename(archive)
    missing = {d for d in db_ids if not (root / d).exists()}
    if missing:
        with zipfile.ZipFile(archive) as z:
            members = [m for m in z.namelist()
                       if m.startswith("database/") and m.split("/")[1] in missing
                       and m.endswith(".sqlite")]
            log.info("Extracting %d test-suite DB files for %d databases", len(members),
                     len(missing))
            z.extractall(root.parent, members)
    return root


def suite_db_files(root: Path, db_id: str) -> list[Path]:
    """All DB variants for ``db_id``, original first (sorted for determinism)."""
    files = sorted((root / db_id).glob("*.sqlite"))
    original = root / db_id / f"{db_id}.sqlite"
    return ([original] if original in files else []) + [f for f in files if f != original]


def db_path(cfg: DictConfig, db_id: str) -> Path:
    """SQLite file for a Spider-dev database id."""
    base = repo_path(cfg.paths.raw_dir) / cfg.data.extract_dir / cfg.data.db_subdir
    return base / db_id / f"{db_id}.sqlite"


# --------------------------------------------------------------------------- #
# Pair loading and sampling
# --------------------------------------------------------------------------- #
def load_set(root: Path, set_name: str) -> list[tuple[dict, dict]]:
    """Load index-aligned (pre, post) records and assert the pairing invariants."""
    pre = json.loads((root / set_name / "questions_pre_perturbation.json").read_text())
    post = json.loads((root / set_name / "questions_post_perturbation.json").read_text())
    if len(pre) != len(post):
        raise ValueError(f"{set_name}: {len(pre)} pre vs {len(post)} post records")

    for i, (a, b) in enumerate(zip(pre, post)):
        # NLQ perturbations rewrite only the question: same DB, same source
        # question, same gold SQL. If any of these fail, the pair is not a twin.
        if (a["db_id"], a["q_id_spider_dev"], a["query"]) != (
            b["db_id"], b["q_id_spider_dev"], b["query"]
        ):
            raise ValueError(f"{set_name}[{i}] is not a valid clean/perturbed twin")
    return list(zip(pre, post))


def sample_pairs(
    cfg: DictConfig, root: Path
) -> tuple[list[Pair], dict[str, Example]]:
    """Sample pairs per set (seeded) and return them with their unique examples."""
    rng = random.Random(cfg.seed)
    pairs: list[Pair] = []
    examples: dict[str, Example] = {}

    for set_name in cfg.data.perturbation_sets:
        records = list(enumerate(load_set(root, set_name)))
        rng.shuffle(records)

        chosen, seen_qids = [], set()
        for idx, (pre, post) in records:
            qid = pre["q_id_spider_dev"]
            if cfg.data.one_pair_per_question and qid in seen_qids:
                continue
            seen_qids.add(qid)
            chosen.append((idx, pre, post))
            if cfg.data.pairs_per_set and len(chosen) >= cfg.data.pairs_per_set:
                break

        for idx, pre, post in sorted(chosen, key=lambda r: r[0]):
            clean = Example(make_uid(pre["db_id"], pre["question"]), pre["db_id"],
                            pre["question"], pre["query"])
            pert = Example(make_uid(post["db_id"], post["question"]), post["db_id"],
                           post["question"], post["query"])
            examples[clean.uid] = clean
            examples[pert.uid] = pert
            pairs.append(Pair(f"{set_name}:{idx}", set_name, pre["q_id_spider_dev"],
                              pre["db_id"], clean.uid, pert.uid))
        log.info("%s: %d eligible records -> %d pairs", set_name, len(records), len(chosen))

    return pairs, examples


def to_rows(items) -> list[dict]:
    return [asdict(x) for x in items]
