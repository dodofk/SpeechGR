from speechgr.utils.docid import (
    extract_cluster_token,
    extract_docid_tokens,
    summarize_cluster_balance,
)


def test_extract_docid_tokens_prefers_structured_tokens():
    docid = "<cl_001> <lf1_002> <lf2_003>"
    assert extract_docid_tokens(docid) == ["<cl_001>", "<lf1_002>", "<lf2_003>"]
    assert extract_cluster_token(docid) == "<cl_001>"


def test_summarize_cluster_balance_counts_clusters():
    summary = summarize_cluster_balance(
        [
            "<cl_000> <lf1_000> <lf2_000>",
            "<cl_000> <lf1_000> <lf2_001>",
            "<cl_001> <lf1_001> <lf2_000>",
        ]
    )
    assert summary["num_clusters"] == 2
    assert summary["num_docids"] == 3
    assert summary["max_cluster_size"] == 2
    assert summary["min_cluster_size"] == 1
