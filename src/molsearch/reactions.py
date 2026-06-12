"""Combinatorial product SMILES from molecule_id (Nova validator logic)."""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


def get_reaction_info(rxn_id: int, db_path: str | Path) -> tuple | None:
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT smarts, roleA, roleB, roleC FROM reactions WHERE rxn_id = ?", (rxn_id,))
        result = cursor.fetchone()
        conn.close()
        return result
    except Exception as exc:
        logger.error("Error getting reaction info: %s", exc)
        return None


def get_molecules(mol_ids: list[int], db_path: str | Path) -> list:
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        molecules = []
        for mol_id in mol_ids:
            cursor.execute("SELECT smiles, role_mask FROM molecules WHERE mol_id = ?", (mol_id,))
            molecules.append(cursor.fetchone())
        conn.close()
        return molecules
    except Exception as exc:
        logger.error("Error getting molecules: %s", exc)
        return [None] * len(mol_ids)


def combine_triazole_synthons(azide_smiles: str, alkyne_smiles: str) -> str | None:
    try:
        m1 = Chem.RWMol(Chem.MolFromSmiles(azide_smiles))
        m2 = Chem.RWMol(Chem.MolFromSmiles(alkyne_smiles))
        if not m1 or not m2:
            return None

        a1 = next(
            (i for i, atom in enumerate(m1.GetAtoms()) if atom.GetSymbol() == "*" and atom.GetIsotope() == 1),
            None,
        )
        a2 = next(
            (i for i, atom in enumerate(m2.GetAtoms()) if atom.GetSymbol() == "*" and atom.GetIsotope() == 2),
            None,
        )
        if a1 is None or a2 is None:
            return None

        n1 = m1.GetAtomWithIdx(a1).GetNeighbors()[0].GetIdx()
        n2 = m2.GetAtomWithIdx(a2).GetNeighbors()[0].GetIdx()

        combined = Chem.RWMol(m1)
        atom_mapping: dict[int, int] = {}

        for i, atom in enumerate(m2.GetAtoms()):
            if i != a2:
                atom_mapping[i] = combined.AddAtom(atom)

        for bond in m2.GetBonds():
            begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a2 not in (begin_idx, end_idx):
                combined.AddBond(atom_mapping[begin_idx], atom_mapping[end_idx], bond.GetBondType())

        combined.RemoveAtom(a1)
        n1_adj = n1 - (1 if n1 > a1 else 0)
        n2_adj = atom_mapping[n2] - (1 if atom_mapping[n2] > a1 else 0)
        combined.AddBond(n1_adj, n2_adj, Chem.BondType.SINGLE)

        Chem.SanitizeMol(combined)
        return Chem.MolToSmiles(combined)
    except Exception as exc:
        logger.error("Error in triazole synthesis: %s", exc)
        return None


def perform_smarts_reaction(smiles1: str, smiles2: str, smarts: str) -> str | None:
    try:
        rxn = AllChem.ReactionFromSmarts(smarts)
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if not mol1 or not mol2:
            return None
        products = rxn.RunReactants((mol1, mol2))
        return Chem.MolToSmiles(products[0][0]) if products else None
    except Exception as exc:
        logger.error("Error in SMARTS reaction: %s", exc)
        return None


def validate_and_order_reactants(
    smiles1: str,
    smiles2: str,
    role_mask1: int,
    role_mask2: int,
    roleA: int,
    roleB: int,
    smiles3: str | None = None,
    role_mask3: int | None = None,
    roleC: int | None = None,
) -> tuple:
    try:
        if smiles3 is None:
            can_react = (
                ((role_mask1 & roleA) == roleA) and ((role_mask2 & roleB) == roleB)
            ) or (((role_mask1 & roleB) == roleB) and ((role_mask2 & roleA) == roleA))
            if not can_react:
                return None, None
            if ((role_mask1 & roleA) == roleA) and ((role_mask2 & roleB) == roleB):
                return smiles1, smiles2
            return smiles2, smiles1

        can_react_12 = (
            ((role_mask1 & roleA) == roleA) and ((role_mask2 & roleB) == roleB)
        ) or (((role_mask1 & roleB) == roleB) and ((role_mask2 & roleA) == roleA))
        can_react_3 = role_mask3 is not None and roleC is not None and (role_mask3 & roleC) == roleC
        if not can_react_12 or not can_react_3:
            return None, None, None
        if (role_mask1 & roleA) and (role_mask2 & roleB):
            return smiles1, smiles2, smiles3
        return smiles2, smiles1, smiles3
    except Exception as exc:
        logger.error("Error validating reactants: %s", exc)
        return (None, None) if smiles3 is None else (None, None, None)


def react_molecules(rxn_id: int, mol1_id: int, mol2_id: int, db_path: str | Path) -> str | None:
    try:
        reaction_info = get_reaction_info(rxn_id, db_path)
        molecules = get_molecules([mol1_id, mol2_id], db_path)
        if not reaction_info or not all(molecules):
            return None

        smarts, roleA, roleB, _roleC = reaction_info
        (smiles1, role_mask1), (smiles2, role_mask2) = molecules

        reactant1, reactant2 = validate_and_order_reactants(
            smiles1, smiles2, role_mask1, role_mask2, roleA, roleB
        )
        if not reactant1 or not reactant2:
            return None

        if rxn_id == 1:
            return combine_triazole_synthons(reactant1, reactant2)
        return perform_smarts_reaction(reactant1, reactant2, smarts)
    except Exception as exc:
        logger.error("Error reacting molecules %s, %s: %s", mol1_id, mol2_id, exc)
        return None


def react_three_components(
    rxn_id: int,
    mol1_id: int,
    mol2_id: int,
    mol3_id: int,
    db_path: str | Path,
) -> str | None:
    try:
        reaction_info = get_reaction_info(rxn_id, db_path)
        molecules = get_molecules([mol1_id, mol2_id, mol3_id], db_path)
        if not reaction_info or not all(molecules):
            return None

        smarts, roleA, roleB, roleC = reaction_info
        (smiles1, role_mask1), (smiles2, role_mask2), (smiles3, role_mask3) = molecules

        validation_result = validate_and_order_reactants(
            smiles1,
            smiles2,
            role_mask1,
            role_mask2,
            roleA,
            roleB,
            smiles3,
            role_mask3,
            roleC,
        )
        if not all(validation_result):
            return None

        reactant1, reactant2, reactant3 = validation_result

        if rxn_id == 3:
            triazole_cooh = combine_triazole_synthons(reactant1, reactant2)
            if not triazole_cooh:
                return None
            amide_smarts = "[C:1](=O)[OH].[N:2]>>[C:1](=O)[N:2]"
            return perform_smarts_reaction(triazole_cooh, reactant3, amide_smarts)

        if rxn_id == 5:
            suzuki_br_smarts = "[#6:1][Br].[#6:2][B]([OH])[OH]>>[#6:1][#6:2]"
            suzuki_cl_smarts = "[#6:1][Cl].[#6:2][B]([OH])[OH]>>[#6:1][#6:2]"
            intermediate = perform_smarts_reaction(reactant1, reactant2, suzuki_br_smarts)
            if not intermediate:
                return None
            return perform_smarts_reaction(intermediate, reactant3, suzuki_cl_smarts)

        return None
    except Exception as exc:
        logger.error(
            "Error in 3-component reaction %s, %s, %s: %s",
            mol1_id,
            mol2_id,
            mol3_id,
            exc,
        )
        return None


def get_smiles_from_reaction(product_name: str, db_path: str | Path | None = None) -> str | None:
    """Resolve product SMILES from rxn:rxn_id:mol1:mol2[:mol3] like the Nova validator."""
    if db_path is None:
        db_path = os.environ.get("MOLECULES_SQLITE", "")
    db_path = Path(db_path)
    if not db_path.is_file():
        logger.error("Combinatorial DB not found: %s", db_path)
        return None

    try:
        parts = product_name.split(":")
        if len(parts) == 4:
            _, rxn_id, mol1_id, mol2_id = parts
            return react_molecules(int(rxn_id), int(mol1_id), int(mol2_id), db_path)
        if len(parts) == 5:
            _, rxn_id, mol1_id, mol2_id, mol3_id = parts
            return react_three_components(
                int(rxn_id), int(mol1_id), int(mol2_id), int(mol3_id), db_path
            )
        logger.error("Invalid reaction format: %s", product_name)
        return None
    except Exception as exc:
        logger.error("Error in combinatorial reaction %s: %s", product_name, exc)
        return None
