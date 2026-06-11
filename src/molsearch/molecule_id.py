from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ParsedMoleculeID:
    molecule_id: str
    rxn: int
    p1: int
    p2: int | None
    p3: int | None


def parse_molecule_id(molecule_id: str) -> ParsedMoleculeID:
    parts = molecule_id.split(":")
    if len(parts) not in (4, 5) or parts[0] != "rxn":
        raise ValueError(f"Invalid molecule_id format: {molecule_id}")

    rxn = int(parts[1])
    p1 = int(parts[2])

    if len(parts) == 4:
        p2 = int(parts[3])
        p3 = None
    else:
        p2 = int(parts[3])
        p3 = int(parts[4])

    return ParsedMoleculeID(
        molecule_id=molecule_id,
        rxn=rxn,
        p1=p1,
        p2=p2,
        p3=p3,
    )


def add_parsed_columns(df):
    def _safe_parse(value: str) -> ParsedMoleculeID | None:
        try:
            return parse_molecule_id(value)
        except Exception:
            return None

    parsed = df["molecule_id"].astype(str).apply(_safe_parse)
    df = df.copy()
    df["rxn"] = parsed.apply(lambda x: x.rxn if x else None)
    df["p1"] = parsed.apply(lambda x: x.p1 if x else None)
    df["p2"] = parsed.apply(lambda x: x.p2 if x else None)
    df["p3"] = parsed.apply(lambda x: x.p3 if x else None)
    return df
