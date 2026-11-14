"""From a molecular structure to the properties partitioning needs.

Give this module a SMILES string and it returns the descriptors
`vivosim.partitioning` consumes, so a structure can be carried through to a
concentration-time profile.

Requires RDKit, which is an optional dependency::

    pip install rdkit

Honest limits
-------------
Two of the inputs come out of structure well, and two do not.

* **logP** is computed by the Crippen atom-contribution method. It is a decent
  estimate -- typically within about a log unit -- and good enough to tell a
  fat-seeking molecule from a water-soluble one.
* **Molecular weight and polar surface area** are exact properties of the
  structure.
* **pKa** is *not* predicted here. Reliable pKa prediction needs a dedicated
  model, and a wrong pKa silently distorts every partition coefficient for an
  ionizable compound. So this module infers only whether acidic or basic groups
  are *present*, as a hint, and you must supply the pKa value itself.
* **Plasma protein binding** is not predictable from structure to the accuracy
  a PBPK model needs. It is measured.

So this is a starting point for a simulation, not a replacement for
characterising the molecule.
"""

from __future__ import annotations

from dataclasses import dataclass

from .partitioning import IonizationClass, MolecularProperties

__all__ = ["StructureDescriptors", "describe_smiles", "properties_from_smiles"]

# Substructures used only to flag that an ionizable group is present, so the
# caller is prompted to supply a pKa rather than silently accepting the
# neutral-compound default.
_ACID_PATTERNS = (
    ("carboxylic acid", "[CX3](=O)[OX2H1]"),
    ("sulfonic acid", "[SX4](=O)(=O)[OX2H1]"),
    ("phosphoric acid", "[PX4](=O)([OX2H1])[OX2H1]"),
    ("tetrazole", "c1nnn[nH]1"),
)
_BASE_PATTERNS = (
    ("aliphatic amine", "[NX3;H2,H1,H0;!$(N[!#6]);!$(N=*);!$(N#*);!$(N-a)]"),
    ("amidine", "[NX3][CX3]=[NX2]"),
    ("guanidine", "[NX3][CX3](=[NX2])[NX3]"),
)


@dataclass(frozen=True)
class StructureDescriptors:
    """Descriptors computed from a molecular structure.

    Attributes:
        smiles: The canonical SMILES of the parsed structure.
        molecular_weight: Exact molar mass in g/mol.
        logp: Crippen estimate of the octanol/water partition coefficient.
        tpsa: Topological polar surface area in square angstroms.
        hydrogen_bond_donors: Count of hydrogen bond donors.
        hydrogen_bond_acceptors: Count of hydrogen bond acceptors.
        rotatable_bonds: Count of rotatable bonds.
        acidic_groups: Names of acidic groups detected, if any.
        basic_groups: Names of basic groups detected, if any.
    """

    smiles: str
    molecular_weight: float
    logp: float
    tpsa: float
    hydrogen_bond_donors: int
    hydrogen_bond_acceptors: int
    rotatable_bonds: int
    acidic_groups: tuple[str, ...] = ()
    basic_groups: tuple[str, ...] = ()

    @property
    def likely_ionization(self) -> IonizationClass:
        """The ionization class the detected groups suggest.

        A hint only. A molecule carrying both acidic and basic groups is a
        zwitterion, which this model does not describe properly; such a
        structure is reported as the stronger influence on partitioning, the
        base, and should be treated with suspicion.
        """
        if self.basic_groups:
            return "base"
        if self.acidic_groups:
            return "acid"
        return "neutral"

    @property
    def is_ionizable(self) -> bool:
        """Whether any ionizable group was detected, meaning a pKa is needed."""
        return bool(self.acidic_groups or self.basic_groups)


def describe_smiles(smiles: str) -> StructureDescriptors:
    """Compute descriptors from a SMILES string.

    Args:
        smiles: The structure, as SMILES.

    Returns:
        The computed descriptors.

    Raises:
        ImportError: If RDKit is not installed.
        ValueError: If the SMILES cannot be parsed.
    """
    chem, descriptors, crippen = _import_rdkit()

    mol = chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"could not parse SMILES {smiles!r}")

    acidic = tuple(
        label for label, smarts in _ACID_PATTERNS if _has_substructure(chem, mol, smarts)
    )
    basic = tuple(
        label for label, smarts in _BASE_PATTERNS if _has_substructure(chem, mol, smarts)
    )

    return StructureDescriptors(
        smiles=chem.MolToSmiles(mol),
        molecular_weight=descriptors.MolWt(mol),
        logp=crippen.MolLogP(mol),
        tpsa=descriptors.TPSA(mol),
        hydrogen_bond_donors=descriptors.NumHDonors(mol),
        hydrogen_bond_acceptors=descriptors.NumHAcceptors(mol),
        rotatable_bonds=descriptors.NumRotatableBonds(mol),
        acidic_groups=acidic,
        basic_groups=basic,
    )


def properties_from_smiles(
    smiles: str,
    fraction_unbound_plasma: float,
    *,
    name: str | None = None,
    pka: float | None = None,
    ionization: IonizationClass | None = None,
    blood_plasma_ratio: float = 1.0,
    logp: float | None = None,
) -> MolecularProperties:
    """Molecular properties for partitioning, from a structure plus measurements.

    Args:
        smiles: The structure, as SMILES.
        fraction_unbound_plasma: Measured unbound fraction in plasma. Required,
            because it is not predictable from structure.
        name: Compound label. Defaults to the canonical SMILES.
        pka: Acid dissociation constant. Required if the structure carries an
            ionizable group, since it is not predicted here.
        ionization: Override the ionization class inferred from the structure.
        blood_plasma_ratio: Measured whole blood to plasma ratio.
        logp: Override the computed logP with a measured value, which is always
            preferable when available.

    Returns:
        `MolecularProperties` ready for `predict_partition_coefficients`.

    Raises:
        ImportError: If RDKit is not installed.
        ValueError: If the SMILES cannot be parsed, or an ionizable structure
            is given without a pKa.
    """
    descriptors = describe_smiles(smiles)
    resolved_class = ionization if ionization is not None else descriptors.likely_ionization

    if resolved_class != "neutral" and pka is None:
        groups = ", ".join(descriptors.acidic_groups + descriptors.basic_groups) or resolved_class
        raise ValueError(
            f"{descriptors.smiles} carries an ionizable group ({groups}), so its pKa "
            "determines how it partitions. pKa is not predicted from structure here -- "
            "supply the measured value as pka=..., or pass ionization='neutral' to "
            "deliberately ignore ionization."
        )

    return MolecularProperties(
        name=name if name is not None else descriptors.smiles,
        logp=descriptors.logp if logp is None else logp,
        fraction_unbound_plasma=fraction_unbound_plasma,
        ionization=resolved_class,
        pka=pka,
        blood_plasma_ratio=blood_plasma_ratio,
        molecular_weight=descriptors.molecular_weight,
    )


def _has_substructure(chem, mol, smarts: str) -> bool:
    pattern = chem.MolFromSmarts(smarts)
    return pattern is not None and mol.HasSubstructMatch(pattern)


def _import_rdkit():
    """Import RDKit, with an actionable message when it is missing."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "reading molecular structures requires RDKit, which vivosim does not "
            "install by default. Install it with `pip install rdkit`, or skip this "
            "module and supply partition coefficients directly to Compound."
        ) from exc
    return Chem, Descriptors, Crippen
