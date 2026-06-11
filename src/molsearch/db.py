from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

MOLECULE_ID_CANDIDATES = ["molecule_id", "mol_id", "id", "molecule", "name"]
SMILES_CANDIDATES = ["smiles", "SMILES", "smile"]


def _list_tables(conn: sqlite3.Connection) -> list[str]:
    query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    rows = conn.execute(query).fetchall()
    return [row[0] for row in rows]


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    return [row[1] for row in rows]


def _find_best_table(conn: sqlite3.Connection, tables: list[str]) -> tuple[str, str, str]:
    candidates: list[tuple[str, str, str]] = []
    for table in tables:
        cols = _table_columns(conn, table)
        mol_col = next((c for c in MOLECULE_ID_CANDIDATES if c in cols), None)
        smiles_col = next((c for c in SMILES_CANDIDATES if c in cols), None)
        if mol_col and smiles_col:
            candidates.append((table, mol_col, smiles_col))

    if not candidates:
        detail = {table: _table_columns(conn, table) for table in tables}
        raise ValueError(
            "Could not detect molecule_id/smiles columns. "
            f"Expected molecule id in {MOLECULE_ID_CANDIDATES} and smiles in {SMILES_CANDIDATES}. "
            f"Available schema: {detail}"
        )

    return candidates[0]


def load_molecules(sqlite_path: str | Path) -> pd.DataFrame:
    path = Path(sqlite_path)
    if not path.exists():
        raise FileNotFoundError(f"SQLite file does not exist: {path}")

    conn = sqlite3.connect(path)
    try:
        tables = _list_tables(conn)
        if not tables:
            raise ValueError(f"No tables found in SQLite database: {path}")

        table, mol_col, smiles_col = _find_best_table(conn, tables)
        df = pd.read_sql_query(
            f"SELECT {mol_col} AS molecule_id, {smiles_col} AS smiles FROM '{table}'",
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        raise ValueError("Detected molecule table is empty.")

    df["molecule_id"] = df["molecule_id"].astype(str)
    df["smiles"] = df["smiles"].astype(str)
    return df
