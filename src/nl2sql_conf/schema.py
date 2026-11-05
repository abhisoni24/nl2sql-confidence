"""Serialize a SQLite database schema into the text shown to the model.

We use the database's own ``CREATE TABLE`` statements (with foreign keys) read
from ``sqlite_master``, which is the common zero-shot text-to-SQL format.
"""

from __future__ import annotations

import re
import sqlite3
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def schema_ddl(db_file: str) -> str:
    """Return newline-joined CREATE TABLE statements, whitespace-normalized."""
    con = sqlite3.connect(f"file:{Path(db_file)}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
            "ORDER BY rowid"
        ).fetchall()
    finally:
        con.close()
    # Collapse runs of whitespace inside each statement to keep prompts compact.
    statements = [re.sub(r"\s+", " ", sql).strip() + ";" for (sql,) in rows]
    return "\n".join(statements)
