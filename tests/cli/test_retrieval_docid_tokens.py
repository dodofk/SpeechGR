from speechgr.utils.docid import collect_docid_tokens


def test_collect_docid_tokens_is_canonicalized():
    valid_ids = [
        "<cl_001> <lf1_010> <lf2_002>",
        "<cl_000> <lf1_001> <lf2_001>",
        "<cl_001> <lf1_000> <lf2_000>",
    ]

    tokens = collect_docid_tokens(valid_ids)

    assert tokens == sorted(tokens)
    assert tokens == [
        "<cl_000>",
        "<cl_001>",
        "<lf1_000>",
        "<lf1_001>",
        "<lf1_010>",
        "<lf2_000>",
        "<lf2_001>",
        "<lf2_002>",
    ]
