import json

import numpy as np

from speechgr.docid import HierarchicalDocIdBuilder, HierarchicalDocIdBuilderConfig


def test_hierarchical_docid_builder_writes_expected_artifacts(tmp_path):
    config = HierarchicalDocIdBuilderConfig(
        num_coarse_clusters=2,
        leaf1_size=4,
        leaf2_size=4,
        seed=7,
    )
    builder = HierarchicalDocIdBuilder(config)

    doc_ids = ["doc-a", "doc-b", "doc-c", "doc-d"]
    embeddings = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.2, 1.1],
            [2.0, 2.0, 0.0],
            [2.2, 1.9, 0.1],
        ],
        dtype=np.float32,
    )

    result = builder.build(doc_ids, embeddings)
    result.write_artifacts(tmp_path)

    assert result.diagnostics["num_documents"] == 4
    assert result.diagnostics["collision_count"] == 0
    assert sorted(result.docid_map) == sorted(doc_ids)
    assert len(result.valid_paths) == len(doc_ids)

    docid_map = json.loads((tmp_path / "docid_map.json").read_text())
    cluster_members = json.loads((tmp_path / "cluster_members.json").read_text())
    valid_paths = json.loads((tmp_path / "valid_paths.json").read_text())
    diagnostics = json.loads((tmp_path / "docid_diagnostics.json").read_text())

    assert docid_map["doc-a"]["tokens"][0].startswith("<cl_")
    assert diagnostics["diagnostics"]["num_clusters"] == 2
    assert len(valid_paths) == 4
    assert sum(len(members) for members in cluster_members.values()) == 4


def test_hierarchical_docid_builder_rejects_duplicate_doc_ids():
    builder = HierarchicalDocIdBuilder()

    try:
        builder.build(
            ["doc-a", "doc-a"],
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        )
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:  # pragma: no cover - keeps failure message precise
        raise AssertionError("Expected duplicate doc_ids to raise ValueError")
