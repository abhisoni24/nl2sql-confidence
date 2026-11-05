"""Only tokens entirely before the closing fence are scored as SQL."""

from nl2sql_conf.model import SQLGenerator


class FakeTokenizer:
    """Token ids index into a fixed vocabulary of strings."""
    vocab = ["SELECT", " name", " FROM", " t", "\n```", "```", ";"]

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)


def make_gen():
    gen = SQLGenerator.__new__(SQLGenerator)   # skip model loading
    gen.tokenizer, gen.stop = FakeTokenizer(), "```"
    return gen


def test_stop_token_excluded():
    assert make_gen()._sql_token_span([0, 1, 2, 3, 4]) == (4, True)


def test_semicolon_kept_fence_dropped():
    assert make_gen()._sql_token_span([0, 1, 2, 3, 6, 5]) == (5, True)


def test_no_fence_scores_everything():
    assert make_gen()._sql_token_span([0, 1, 2, 3]) == (4, False)
