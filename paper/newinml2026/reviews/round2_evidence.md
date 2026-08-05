# Round 2 strict evidence audit

## Verdict: NOT OK

The revision resolves the substantive Round 1 evidence problems: unsupported
dataset counts are gone, SpeechDPR is visibly marked as imported and
supervision-mismatched, ablation splits remain unresolved rather than invented,
ranking-loss totals are no longer reconstructed, negative results are locally
calibrated, and the rendered tables are readable. However, the rubric defines
“okay” as requiring every quantitative claim to be in `results_ledger.md`.
One pair of paper claims is still absent from the ledger, so a hard gate
technically remains open.

## Hard-gate failures

### H1. The pretraining deltas are not recorded in the result ledger

Experiments Section 4.3 reports the derived 0-to-10-epoch differences as
\(+2.20\) Hit@1 and \(+4.263\) Hit@20. These values are arithmetically correct:

- \(5.81-3.61=2.20\)
- \(18.452-14.189=4.263\)

They also appear in the thesis's experiment summary. Nevertheless, neither
value occurs in `results_ledger.md`, while hard gate 1 explicitly requires
every quantitative paper claim to be present there.

Exact fix: either add a ledger entry that labels both values as
thesis-summary/derived deltas, with the same unresolved-split caveat as the
pretraining rows, or delete the delta sentence and leave only the three
reported absolute rows. This is a bookkeeping hard failure, not a challenge to
the arithmetic or the underlying thesis evidence.

No other hard-gate failure was found:

- All remaining experimental values are present in the ledger and thesis PDF.
- SpeechDPR values are labeled as reported/imported rather than reproduced.
- Test is assigned only to the headline thesis evaluation; ablation row splits
  remain unresolved.
- No seeds, uncertainty estimates, baseline details, or missing
  hyperparameters are invented.
- Index-free is defined as no document-embedding index or ANN search, and the
  trie is described as constrained decoding.
- The paper makes no state-of-the-art, fully-textless, unwritten-language,
  or unsupported priority claim.
- The official double-blind workshop style is used, the PDF metadata is
  anonymous, and content ends on page 7 before references on the same page,
  within the eight-content-page limit.

## Major concerns

None.

The central comparison is now appropriately contextual rather than presented
as matched. Table 1 separates input, training signal/transcript use, and
retrieval store; its caption discloses the unverified prior-work split. The
ablation prose consistently describes reported scores and unresolved shared
settings instead of causal effects. Each negative experiment includes a
hypothesis, the exact recorded change (or native reported delta), and a local
decision with causal alternatives left open.

## Minor suggestions

1. Table 3 mixes absolute Hit values with ranking-loss deltas under the same
   `Hit@1` and `Hit@20` headers. The delta symbols and caption make the meaning
   recoverable, so this is not a major readability problem. It would be cleaner
   to label the headers `Hit@1 / change` and `Hit@20 / change`, or give ranking
   loss a dedicated `Reported change` column.
2. The thesis describes the ranking-loss decreases as approximate, but the
   table cells print `\(\Delta-0.88\)` and `\(\Delta-1.349\)` without an
   approximation sign. Match the ledger with
   `\(\Delta\approx-0.88\)` and `\(\Delta\approx-1.349\)`.
3. “Showing that distillation is consequential” is slightly stronger than
   necessary for an imported two-row comparison whose original split was not
   independently checked. “The thesis-reported with/without-KD values differ
   substantially” would be maximally neutral.
4. In the ranking-loss paragraph, prefix “convergence is faster” with “the
   thesis reports” so readers do not mistake the qualitative statement for a
   curve or convergence statistic shown in this paper.
5. Figure 1 is now legible and has no text overlap. The fine-tuning and
   inference grouping boxes touch/overlap slightly near the target-DocID and
   inference-unit nodes; a few millimeters of separation would improve polish.
6. Use `Hit@1` and `Hit@20` rather than `H@1` and `H@20` in Table 1 for
   consistency with the other tables and prose.

## Quantitative-claim audit

| Claim group | Evidence/provenance | Audit result |
|---|---|---|
| SpeechGR 5.81 Hit@1 / 18.452 Hit@20 | Thesis Table 4.2, summary, conclusion; ledger headline row | Pass. The thesis prose typo 5.21 is correctly resolved in favor of the repeated 5.81 value. |
| SpeechDPR 19.73 with KD / 0.04 without KD | Imported rows in thesis Table 4.2 and ledger | Pass. Table 1 marks them imported, dense-ANN, supervision-mismatched, and split-unverified. |
| 1.278-point Hit@20 gap | Correct difference of 19.73 and 18.452; recorded in ledger editorial decisions | Pass. It is called a point difference, not a relative percentage. |
| BM25 0.00/0.00 and ASR+BM25 1.50/5.20 | Thesis Table 4.2 and ledger | Pass. The unresolved BM25 input and missing ASR setup are disclosed, and no performance explanation is inferred. |
| Main configuration: layer 22, \(K=500\), 10 pretraining epochs, 6k hours, 15% masking, beam 20 | Thesis Model Configuration and headline ledger row | Pass. The 100-versus-960-hour codebook discrepancy is explicitly disclosed. |
| Unit/BPE table: all seven rows | Thesis Table 4.3 and corresponding ledger rows | Pass. Same-\(K\) comparisons are limited to \(K=500\) and \(K=1000\); row-level split/pretraining ambiguity is in the caption and prose. |
| Pretraining rows: 0/3/10 epochs with 3.61/14.189, 4.54/15.73, 5.81/18.452 | Thesis Table 4.4 and ledger | Pass for absolute values. Split and shared-setting ambiguity are disclosed. |
| Pretraining deltas +2.20/+4.263 | Correct arithmetic and thesis experiment summary | **Hard-gate bookkeeping fail:** absent from `results_ledger.md`. |
| Identifier rows: string 3.61/14.19; atomic 4.54/13.69 | Thesis Identifier Design subsection and ledger | Pass. Presented as a metric trade-off without significance claims. |
| Ranking-loss baseline 3.61/14.189 and changes -0.88/-1.349 | Thesis Ranking Loss subsection and ledger | Pass. Absolute intervention scores are no longer reconstructed; add approximation symbols as a minor precision fix. |
| Q-Former \(L=17,Q=1\): 3.61/14.189 to 2.73/12.584 | Thesis Q-Former subsection and ledger | Pass. Mechanism is explicitly not isolated. |
| VAD: 3.61/14.189 to 3.27/12.97 | Thesis VAD subsection and ledger | Pass. Deduplication interaction is labeled possible and untested. |
| “One English benchmark,” no repeated-run uncertainty, and no dataset-size counts | Thesis scope and omissions | Pass. Unsupported exact split/corpus counts have been removed. |

Once H1 is resolved, the evidence/statistics review would be **OK**; the
remaining items are presentation refinements rather than major concerns.
