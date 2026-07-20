"""Catalog/dispatch consistency for the dataset loaders (offline-safe)."""

from __future__ import annotations

import pytest

from judge_bias.dataset.loaders import (
    DATASET_CATALOG,
    list_datasets,
    load_dataset_items,
)

OFFLINE = {"synthetic", "synthetic-offensive", "synthetic-defensive",
           "synthetic-anchor"}
# Minimal open-weight set: primary + control + objective anchor + probe.
EXPECTED_OPEN_SETS = {"cyberseceval", "mt-bench", "objective-anchor",
                      "cti-bench", "wmdp"}


def test_catalog_lists_known_open_datasets():
    assert EXPECTED_OPEN_SETS <= set(DATASET_CATALOG)
    # every entry advertises a role, HF id and license
    for info in list_datasets().values():
        assert info["role"] in {"primary", "anchor", "control", "probe"}
        assert info["hf_id"] and info["license"]


def test_every_catalog_entry_is_dispatchable():
    # a registered loader must resolve - never the "unknown dataset" error. It may
    # then load (offline set, or a dataset already cached/cloned locally) or fail for
    # another reason (missing datasets extra, gated or network-only set); both are OK.
    for name in DATASET_CATALOG:
        if name in OFFLINE:
            assert load_dataset_items(name, n=2)
        else:
            try:
                load_dataset_items(name, limit=2)
            except Exception as ei:  # noqa: BLE001 - any non-dispatch error is acceptable here
                assert "unknown dataset" not in str(ei)


def test_unknown_dataset_rejected():
    with pytest.raises(ValueError, match="unknown dataset"):
        load_dataset_items("does-not-exist")


def test_roles_cover_each_study_slot():
    roles = {info["role"] for info in DATASET_CATALOG.values()}
    assert {"primary", "anchor", "control", "probe"} <= roles
