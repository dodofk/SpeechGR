# Round 3 focused evidence recheck

## Verdict: OK

The latest draft passes all hard gates in `review_rubric.md`, and no major
evidence, statistics, or readability concern remains.

## Hard-gate failures

None.

- Every experimental number in the paper is now present in
  `results_ledger.md`. In particular, the ledger explicitly records the
  \(+2.20\) Hit@1 and \(+4.263\) Hit@20 pretraining differences as transparent
  deltas derived from the thesis rows, not as a separate run.
- The ranking-loss intervention preserves the thesis's native reporting:
  \(\Delta\approx-0.88\) Hit@1 and
  \(\Delta\approx-1.349\) Hit@20. No absolute intervention score is
  reconstructed.
- SpeechGR's 5.81/18.452 headline result is tied to the thesis-reported test
  evaluation. Ablation rows do not receive invented splits or shared
  pretraining settings.
- SpeechDPR's 19.73 and 0.04 Hit@20 values are marked as imported prior-work
  results, with the split unverified and the knowledge-distilled setting
  explicitly not supervision-matched to SpeechGR.
- The BM25 input inconsistency, missing ASR configuration, 100-versus-960-hour
  codebook discrepancy, absent repeated-run uncertainty, and missing
  reproduction details are disclosed rather than filled in.
- “Index-free” is precisely defined as no external document-embedding index or
  ANN search. The DocID trie is correctly described as a constrained-decoding
  structure.
- The paper makes no state-of-the-art, fully-textless, unwritten-language,
  or unsupported priority claim.
- The official double-blind workshop style remains in use, PDF metadata is
  anonymous, and content ends on page 7 before the references, within the
  eight-content-page limit.

## Major concerns

None.

The main table now makes input modality, training signal/transcript use,
retrieval store, result provenance, and supervision mismatch visible. All
three result tables are legible at normal viewing size. Each negative
experiment states its hypothesis, reproduces the exact thesis value or native
reported delta, and ends with a local design decision while leaving untested
mechanisms explicitly unresolved. The latest PDF has no broken references,
clipped tables, overlapping text, placeholder result values, or material
overfull boxes.

## Optional minor note

Table 3 necessarily mixes absolute Hit values with explicitly marked
ranking-loss deltas under the same metric columns. The approximation and delta
symbols plus the caption make this unambiguous, so it is acceptable for the
current submission and does not block the verdict.
