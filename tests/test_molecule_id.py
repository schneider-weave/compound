import pandas as pd

from molsearch.molecule_id import parse_molecule_id


def test_parse_two_param_reaction_id() -> None:
    parsed = parse_molecule_id("rxn:2:166916:113926")
    assert parsed.rxn == 2
    assert parsed.p1 == 166916
    assert parsed.p2 == 113926
    assert parsed.p3 is None


def test_parse_three_param_reaction_id() -> None:
    parsed = parse_molecule_id("rxn:3:111:222:333")
    assert parsed.rxn == 3
    assert parsed.p1 == 111
    assert parsed.p2 == 222
    assert parsed.p3 == 333


def test_non_rxn_molecule_id_is_handled_in_dataframe_parse() -> None:
    from molsearch.molecule_id import add_parsed_columns

    df = pd.DataFrame([{"molecule_id": "123", "smiles": "CCO"}])
    parsed_df = add_parsed_columns(df)
    assert parsed_df["rxn"].isna().all()
