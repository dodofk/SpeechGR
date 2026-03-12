from speechgr.docid import analyze_docid_map


def test_analyze_docid_map_reports_cluster_skew():
    report = analyze_docid_map(
        {
            "doc-a": {"docid": "<cl_000> <lf1_000> <lf2_000>", "cluster": "<cl_000>"},
            "doc-b": {"docid": "<cl_000> <lf1_000> <lf2_001>", "cluster": "<cl_000>"},
            "doc-c": {"docid": "<cl_001> <lf1_000> <lf2_000>", "cluster": "<cl_001>"},
        }
    )

    assert report.num_documents == 3
    assert report.num_clusters == 2
    assert report.max_cluster_size == 2
    assert report.min_cluster_size == 1
    assert report.singleton_cluster_count == 1


def test_analyze_docid_map_reports_true_median_for_even_cluster_counts():
    report = analyze_docid_map(
        {
            "doc-a": {"docid": "<cl_000> <lf1_000> <lf2_000>", "cluster": "<cl_000>"},
            "doc-b": {"docid": "<cl_000> <lf1_000> <lf2_001>", "cluster": "<cl_000>"},
            "doc-c": {"docid": "<cl_001> <lf1_000> <lf2_000>", "cluster": "<cl_001>"},
            "doc-d": {"docid": "<cl_002> <lf1_000> <lf2_000>", "cluster": "<cl_002>"},
            "doc-e": {"docid": "<cl_002> <lf1_000> <lf2_001>", "cluster": "<cl_002>"},
            "doc-f": {"docid": "<cl_002> <lf1_000> <lf2_002>", "cluster": "<cl_002>"},
            "doc-g": {"docid": "<cl_003> <lf1_000> <lf2_000>", "cluster": "<cl_003>"},
            "doc-h": {"docid": "<cl_003> <lf1_000> <lf2_001>", "cluster": "<cl_003>"},
            "doc-i": {"docid": "<cl_003> <lf1_000> <lf2_002>", "cluster": "<cl_003>"},
            "doc-j": {"docid": "<cl_003> <lf1_000> <lf2_003>", "cluster": "<cl_003>"},
        }
    )

    # Cluster sizes are [1, 2, 3, 4], so the true median is 2.5.
    assert report.median_cluster_size == 2.5
