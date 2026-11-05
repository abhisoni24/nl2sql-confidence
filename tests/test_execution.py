"""Execution-match rules on a tiny in-memory-built SQLite file."""

import sqlite3

import pytest

from nl2sql_conf.execution import QueryError, execute, gold_is_ordered, results_match


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "toy.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE singer(id INTEGER, name TEXT, age REAL);"
        "INSERT INTO singer VALUES (1,'Ann',30.0),(2,'Bo',40.0),(3,'Cy',20.0);"
    )
    con.commit()
    con.close()
    return path


def test_column_order_ignored(db):
    gold = execute(db, "SELECT name, age FROM singer", 5)
    pred = execute(db, "SELECT age, name FROM singer", 5)
    assert results_match(gold, pred, ordered=False)


def test_row_order_matters_only_with_order_by(db):
    asc = execute(db, "SELECT name FROM singer ORDER BY age", 5)
    desc = execute(db, "SELECT name FROM singer ORDER BY age DESC", 5)
    assert results_match(asc, desc, ordered=False)
    assert not results_match(asc, desc, ordered=True)
    assert gold_is_ordered("select name from singer order  by age")


def test_float_rounding(db):
    gold = execute(db, "SELECT AVG(age) FROM singer", 5)            # 30.0
    assert results_match(gold, [(30.00001,)], ordered=False, decimals=4)
    assert not results_match(gold, [(30.1,)], ordered=False, decimals=4)


def test_duplicates_count(db):
    assert not results_match([(1,), (1,)], [(1,)], ordered=False)


def test_wrong_column_count(db):
    gold = execute(db, "SELECT name FROM singer", 5)
    pred = execute(db, "SELECT name, age FROM singer", 5)
    assert not results_match(gold, pred, ordered=False)


def test_bad_sql_raises(db):
    with pytest.raises(QueryError):
        execute(db, "SELECT nope FROM singer", 5)


def test_read_only(db):
    with pytest.raises(QueryError):
        execute(db, "DELETE FROM singer", 5)
