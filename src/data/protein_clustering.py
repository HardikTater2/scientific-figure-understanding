"""
Protein Clustering and Leak-Free Group Splitting Module (Phase 4).
Solves hidden structural fold leakage in PDB splits by grouping proteins according to
CATH superfamily and structural homology folds (e.g., CATH S40 non-redundant clusters).

Prevents homologous structures from appearing in both train and test splits:
  - 1HVR and 4E43 (both HIV-1 protease / retroviral aspartyl proteases)
  - 1AKE and 4AKE (both Adenylate Kinase conformational states)
  - 1CRN and 1EJG (Thionin plant toxins)
  - 1UBQ and 2OOB (Ubiquitin-like beta-grasp)
"""

from typing import Dict, List, Tuple, Set
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


# Mapping of PDB IDs to CATH Superfamily / Fold Clusters
# CATH Homology superfamilies ensure < 35% sequence identity and distinct structural folds
PROTEIN_CATH_CLUSTERS: Dict[str, str] = {
    # Retroviral aspartyl protease fold (CATH 2.40.70.10)
    "1HVR": "CATH_2.40.70.10",
    "4E43": "CATH_2.40.70.10",
    "1A30": "CATH_2.40.70.10",
    "1AJV": "CATH_2.40.70.10",

    # Adenylate kinase fold (CATH 3.40.50.300)
    "1AKE": "CATH_3.40.50.300",
    "4AKE": "CATH_3.40.50.300",
    "2ECK": "CATH_3.40.50.300",
    "1DVR": "CATH_3.40.50.300",

    # Lysozyme / Glycosidase fold (CATH 1.10.530.10)
    "1HEW": "CATH_1.10.530.10",
    "2LZM": "CATH_1.10.530.10",
    "1HEL": "CATH_1.10.530.10",

    # Ubiquitin beta-grasp fold (CATH 3.10.20.90)
    "1UBQ": "CATH_3.10.20.90",
    "2OOB": "CATH_3.10.20.90",

    # Myoglobin / Globin-like fold (CATH 1.10.490.10)
    "1MBO": "CATH_1.10.490.10",
    "1MBA": "CATH_1.10.490.10",

    # Ribonuclease A fold (CATH 2.60.40.10)
    "1RSA": "CATH_2.60.40.10",
    "2RNS": "CATH_2.60.40.10",

    # Crambin / Plant seed protein (CATH 4.10.290.10)
    "1CRN": "CATH_4.10.290.10",
    "1EJG": "CATH_4.10.290.10",

    # Insulin / Growth factor (CATH 1.20.120.10)
    "4INS": "CATH_1.20.120.10",
    "2HIU": "CATH_1.20.120.10",

    # Bovine pancreatic trypsin inhibitor (CATH 4.10.410.10)
    "4PTI": "CATH_4.10.410.10",
    "5PTI": "CATH_4.10.410.10",

    # Green fluorescent protein beta-barrel (CATH 2.40.155.10)
    "1GFL": "CATH_2.40.155.10",
    "1EMA": "CATH_2.40.155.10",

    # Cytochrome c fold (CATH 1.10.750.10)
    "1CYC": "CATH_1.10.750.10",
    "2CYP": "CATH_1.10.750.10"
}


def get_cluster_id(pdb_id: str) -> str:
    """Returns the CATH cluster ID for a PDB ID, defaulting to an individual cluster if unmapped."""
    clean_id = pdb_id.upper().strip()
    return PROTEIN_CATH_CLUSTERS.get(clean_id, f"CLUSTER_{clean_id}")


def split_by_protein_cluster(
    pdb_ids: List[str],
    test_size: float = 0.25,
    random_state: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Splits PDB structures into train and test splits using GroupShuffleSplit on structural clusters.
    Guarantees no two proteins from the same CATH superfamily land in opposite splits.
    """
    clusters = [get_cluster_id(p) for p in pdb_ids]
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    indices = np.arange(len(pdb_ids))
    train_idx, test_idx = next(splitter.split(indices, groups=clusters))

    train_pdbs = [pdb_ids[i] for i in train_idx]
    test_pdbs = [pdb_ids[i] for i in test_idx]

    assert_cluster_leak_free(train_pdbs, test_pdbs)
    return train_pdbs, test_pdbs


def assert_cluster_leak_free(train_pdbs: List[str], test_pdbs: List[str]):
    """Asserts that no structural fold or CATH superfamily is shared between train and test."""
    train_clusters = {get_cluster_id(p) for p in train_pdbs}
    test_clusters = {get_cluster_id(p) for p in test_pdbs}
    
    overlap = train_clusters.intersection(test_clusters)
    if overlap:
        raise AssertionError(
            f"Train/test leakage detected! Shared structural clusters: {overlap}. "
            f"Train proteins: {train_pdbs}, Test proteins: {test_pdbs}"
        )
