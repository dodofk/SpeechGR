import torch

from speechgr.docid import TfidfPassageEmbeddingConfig, build_tfidf_passage_embeddings


def test_build_tfidf_passage_embeddings_writes_expected_shape(tmp_path):
    cache_path = tmp_path / "corpus_mimi.pt"
    torch.save(
        {
            "doc-a": {"codes": torch.tensor([0, 1, 1, 2], dtype=torch.long)},
            "doc-b": {"codes": torch.tensor([2, 2, 3, 3], dtype=torch.long)},
            "doc-c": {"codes": torch.tensor([4, 4, 4, 5], dtype=torch.long)},
        },
        cache_path,
    )

    result = build_tfidf_passage_embeddings(
        cache_path,
        config=TfidfPassageEmbeddingConfig(
            vocab_size=8,
            projection_dim=4,
            seed=7,
        ),
    )

    assert result.doc_ids == ["doc-a", "doc-b", "doc-c"]
    assert result.embeddings.shape == (3, 4)
    assert result.diagnostics["num_documents"] == 3
    assert result.diagnostics["output_embedding_dim"] == 4
