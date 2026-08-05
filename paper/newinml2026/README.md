# SpeechGR for NewInML 2026

This directory contains the anonymous workshop-paper conversion of the
completed SpeechGR master's thesis.

Evidence policy:

- Use thesis results from `../../master_thesis_v0.pdf`, plus the explicitly
  labelled BM25 reproduction below.
- Keep imported prior-work results visibly separate from thesis experiments.
- Record every reported number and ambiguity in `results_ledger.md`.
- Do not add post-thesis neural-model results.

BM25 reproduction:

From the repository root:

```bash
.venv/bin/python scripts/evaluate_bm25.py \
  --splits validation test verified_test \
  --output output/metrics/bm25_slue_sqa5_text.json
```

This uses only the local text metadata: 2,382 test questions, 408
verified-test questions, and 15,883 unique passages. It does not download or
decode audio. The official full SQA-5 package is about 118 GB to download, so
it is unnecessary for this baseline.

Build:

```bash
make
```

The review build uses the official NeurIPS 2026 style with the
`dblblindworkshop` option.

Review status:

- The BM25/literature revision passed all three round-four reviews: framing and
  literature, evidence and reproducibility, and manuscript/layout.
- The title and SVQ/MSEB revision passed two focused round-five reviews:
  literature accuracy and title/manuscript claim precision.
- The query-modality contribution revision passed both round-six reviews:
  literature accuracy and manuscript framing. BM25 is now evaluation context,
  while the contribution is scoped to spoken-query generative retrieval.
- The IRGen/GENIUS task-positioning rewrite passed all three round-seven
  reviews: literature precision, manuscript coherence, and evidence/layout.
  The final wording treats IRGen as a structural analogue, GENIUS as
  instruction-conditioned and CLIP-aligned, and SLUE-SQA-5 as a directed
  question-to-answer-containing-passage relation without claiming semantic
  understanding.
- The reviewed build uses all eight permitted content pages; the conclusion and
  references begin on page 8, and references continue through page 10.
- Reviewed PDF SHA-256:
  `ab5f6589a8835c65175e6dcd510a9ce49c974b927db2af7e6c1018ee5e67dcb7`.
