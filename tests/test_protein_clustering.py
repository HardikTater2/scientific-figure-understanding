"""
Unit tests for protein structural cluster splitting and leakage prevention (Phase 4).
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.data.protein_clustering import (
    PROTEIN_CATH_CLUSTERS,
    get_cluster_id,
    split_by_protein_cluster,
    assert_cluster_leak_free
)


def test_cath_cluster_mapping():
    assert get_cluster_id("1hvr") == "CATH_2.40.70.10"
    assert get_cluster_id("4e43") == "CATH_2.40.70.10"
    assert get_cluster_id("1ake") == "CATH_3.40.50.300"
    assert get_cluster_id("4ake") == "CATH_3.40.50.300"
    assert get_cluster_id("1HVR") == get_cluster_id("4E43")
    assert get_cluster_id("1AKE") == get_cluster_id("4AKE")


def test_cluster_split_prevents_leakage():
    pdb_pool = [
        "1HVR", "4E43", "1A30", "1AJV",
        "1AKE", "4AKE", "2ECK", "1DVR",
        "1HEW", "2LZM", "1HEL",
        "1UBQ", "2OOB",
        "1MBO", "1MBA",
        "1RSA", "2RNS",
        "1CRN", "1EJG",
        "4INS", "2HIU",
        "4PTI", "5PTI",
        "1GFL", "1EMA",
        "1CYC", "2CYP"
    ]

    train_pdbs, test_pdbs = split_by_protein_cluster(pdb_pool, test_size=0.30, random_state=42)

    # 1. Train and test sets must be disjoint
    assert set(train_pdbs).isdisjoint(set(test_pdbs))

    # 2. HIV-1 proteases (1HVR and 4E43) must not be split across train and test
    hiv_in_train = any(p in train_pdbs for p in ["1HVR", "4E43"])
    hiv_in_test = any(p in test_pdbs for p in ["1HVR", "4E43"])
    assert not (hiv_in_train and hiv_in_test), "HIV-1 proteases leaked across train and test!"

    # 3. Adenylate kinases (1AKE and 4AKE) must not be split across train and test
    ake_in_train = any(p in train_pdbs for p in ["1AKE", "4AKE"])
    ake_in_test = any(p in test_pdbs for p in ["1AKE", "4AKE"])
    assert not (ake_in_train and ake_in_test), "Adenylate kinases leaked across train and test!"

    # 4. Cluster-level assertion function passes
    assert_cluster_leak_free(train_pdbs, test_pdbs)


def test_cluster_assertion_catches_intentional_leak():
    # If 1HVR is in train and 4E43 is in test, assertion must raise AssertionError
    leaky_train = ["1HVR", "1UBQ"]
    leaky_test = ["4E43", "1AKE"]

    with pytest.raises(AssertionError, match="Train/test leakage detected"):
        assert_cluster_leak_free(leaky_train, leaky_test)
