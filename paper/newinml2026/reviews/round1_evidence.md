# Round 1 evidence and statistics review

## Verdict: not okay

The paper has a coherent feasibility claim and the headline SpeechGR numbers
are handled more carefully than in the thesis. However, it does not yet meet
the review rubric. One thesis-only evidence hard gate fails, and several major
issues remain in the supervision comparison, ablation interpretation, and
rendered artifact.

The defensible central claim is:

> In the thesis-reported test evaluation, SpeechGR obtains 5.81 Hit@1 and
> 18.452 Hit@20. Its Hit@20 is 1.278 points below the 19.73 value imported for
> knowledge-distilled SpeechDPR, under different training supervision and
> retrieval mechanisms. Single-run ablations in the thesis suggest useful
> design choices, but their per-row evaluation splits and some shared settings
> are not fully recorded.

## Hard-gate failures

### H1. Exact dataset sizes are not evidence from the thesis PDF

The paper reports 46,186 training queries, 1,939 validation queries, 2,382 test
queries, a 408-query verified-test subset, and 15,883 candidate passages.
None of these counts appears in the thesis PDF or active thesis experiment
source. They also do not appear as evidence rows in `results_ledger.md`.
They appear to come from a later local metadata count. Under the agreed
thesis-PDF-only boundary, this fails the thesis-only evidence gate.

Affected claims:

- Experiments, Data and evaluation: all five counts.
- Limitations: “a corpus of 15,883 passages.”

Exact fix: remove the five counts from the workshop paper. Replace the setup
sentence with: “The thesis defines train, validation, test, and verified-test
splits over SLUE-SQA-5 and indexes the linked Spoken Wikipedia passages, but
does not report their sizes.” In Limitations, replace the exact corpus count
with “the linked SLUE-SQA-5 passage collection.” If the authors later broaden
the evidence policy to allow dataset-file accounting, add a separate data
provenance ledger with file paths, row-count commands, and duplicate/filtering
rules; do not present those counts as thesis-PDF evidence.

## Major concerns, ranked

### M1. The main comparison table does not expose the supervision difference clearly enough

The table's “Transcript / extra sup.” column conflates three different
concepts: query modality, task-time transcripts, and knowledge distillation.
“No transcript; KD” for SpeechDPR does not tell the reader that the thesis
describes its distillation as coming from an unsupervised ASR model and a text
retriever. “None” for SpeechGR also hides the text-pretrained Flan-T5
initialization and 6,000-hour acoustic-unit pretraining. The caption warns that
the settings are not matched, but the cells do not show why.

The caption also calls the entire table “results on the ... test split.” The
thesis labels its main table as test-set evaluation, but the two SpeechDPR
values are imported and their original evaluation split has not been
independently verified. The present wording visually implies a matched
same-split comparison.

Exact fix:

- Replace the combined column with separate columns such as `Query at
  inference`, `Training signal / initialization`, `Retrieval store`, and
  `Source / split`.
- SpeechGR should say, compactly, `speech`, `text-pretrained Flan-T5 +
  LibriLight unit pretraining; no task transcripts`, `parametric + DocID trie`,
  and `this thesis; test`.
- SpeechDPR with KD should say `speech`, `KD from ASR/text-retrieval teachers
  (as described by thesis)`, `dense ANN`, and `prior work; split not
  independently verified`.
- SpeechDPR without KD should be labeled analogously.
- Change `text (unresolved)` to `input unresolved in thesis`; a question mark
  reads as an unfinished placeholder.
- Caption the table as a “thesis-assembled comparison,” not a fully matched
  benchmark table.

### M2. “Controlled experiments” and “improve” overstate the ablation evidence

The paper commendably says that ablation splits are not unambiguously stated.
That caveat is inconsistent with the abstract and introduction calling the
studies “controlled experiments/evidence.” The thesis says ablations initially
use development data, that promising configurations may advance to test, and
that some granularity rows omit speech pretraining without naming them. The
10-epoch pretraining row exactly equals the headline test result. Therefore,
the paper cannot establish that all rows use the same split or all non-varied
settings.

Affected wording includes:

- Abstract: “controlled experiments identify” and “units ... and pretraining
  improve retrieval.”
- Introduction contribution 3: “controlled evidence.”
- Results: “making pretraining the strongest positive intervention.”
- Conclusion: “identifies speech-unit pretraining as beneficial.”

Exact fix: consistently describe these as “thesis-reported single-run
ablations.” Replace causal or controlled language with observed-score
language, e.g. “Later-layer units and longer speech-unit pretraining are
associated with higher reported scores in the tested configurations.” Keep the
split caveat in both table captions and the first analysis paragraph, and do
not call the 0-to-10 comparison a test-set gain.

### M3. Ranking-loss totals should not be reconstructed as result values

The active thesis PDF reports a baseline of 3.61/14.189 and approximate drops
of 0.88/1.349; it does not print the ranking-loss system's absolute scores.
The paper reconstructs approximately 2.73/12.84 and places them in a result
table. The caption discloses the arithmetic and uses approximation symbols,
which is better than presenting them as measured exact values, but it still
turns approximate deltas into apparently observed row scores. The ledger's
additional corroboration from a commented source block is outside the strict
PDF-result boundary.

Exact fix: preserve the thesis's native reporting. In Table 3, show the ranking
row as `reported change: -0.88 / -1.349` rather than absolute Hit@1/Hit@20, or
use em dashes in the absolute-score cells and add a `Reported delta` column.
In prose say: “The thesis reports approximate decreases of 0.88 Hit@1 and
1.349 Hit@20 from the 3.61/14.189 baseline.” Remove “it reaches approximately
2.73/12.84.”

### M4. Several negative-result lessons still imply mechanisms not tested

The requested hypothesis--result--local-lesson pattern is mostly present, and
the Q-Former paragraph appropriately says the mechanism is not isolated.
Other paragraphs remain too causal:

- BPE: “these BPE merges do not preserve the information needed” is an
  information-loss claim not measured by the experiment.
- VAD: “Because ... deduplication already compresses ... the added
  preprocessing is unnecessary” treats an untested explanation as the basis
  for the conclusion.
- Method: BPE “consistently degrades retrieval” is stronger than warranted:
  only two BPE rows have same-\(K\) no-BPE comparators, and pretraining settings
  for the granularity rows are unresolved.

Exact fix:

- BPE local lesson: “The tested BPE settings score below their same-\(K\)
  no-BPE rows, so the final system omits BPE; the experiment does not isolate
  whether the difference comes from information loss, optimization, or an
  unresolved shared setting.”
- VAD local lesson: “The tested VAD preprocessing is not selected because its
  reported score is lower. Interaction with deduplication is a possible but
  untested explanation.”
- Method: replace “consistently degrades” with “the reported BPE variants are
  lower than the available no-BPE comparators.”

### M5. The current PDF fails the rendered-artifact quality gate

The PDF has two visible `Section ??` references on pages 3 and 4 even though
`sec:analysis` now exists in the source. This is likely a stale/single-pass
build, but the reviewed artifact is still broken. Figure 1 also has overlapping
labels and arrows around `cross-entropy`, `gold DocID`, and the trie constraint,
making the inference relationship difficult to read. Hyperlink rectangles are
visible around every citation and internal reference, which is distracting
though not itself an evidence error.

Exact fix:

- Rebuild LaTeX for enough passes to resolve all references and fail the build
  if `undefined references` remains in the log.
- Redraw the lower figure row with the trie above or below the decoder output,
  not horizontally compressed into the same line. Keep one short arrow label
  per edge and ensure the trie-to-decoder constraint arrow does not overlap the
  DocID node.
- Use the template-compatible hidden-link setting so citation boxes do not
  render.
- Re-render every page and visually inspect before the next review.

## Minor concerns

1. The conclusion says the SpeechDPR value is “near” SpeechGR. Prefer the exact
   1.278-point gap because “near” is subjective and the comparison is not
   matched.
2. The paper mixes two and three decimal places. Preserving source precision is
   legitimate, but state once that values are reproduced at thesis precision;
   do not let extra decimals imply greater statistical certainty.
3. The main table is legible and not clipped, but the two full-width tables at
   the top of page 5 make the evidence hierarchy visually flat. After fixing
   the main-table columns, consider reducing Table 2's caption and replacing
   its unused `Panel` column with full-width panel headers.
4. The method's statement that the final system uses no BPE is a reasonable
   editorial resolution from the selected \(K=500\) row, but the thesis Data
   section generically says speech is processed with BPE. Keep this
   inconsistency in the source audit and do not imply it was unambiguous in the
   thesis.
5. The paper has no repeated runs, uncertainty intervals, or significance
   tests. The Limitations section states this well; the abstract and conclusion
   should match that restraint.

## Quantitative-claim audit

| Claim in workshop draft | Thesis-PDF evidence | Status | Required action |
|---|---|---|---|
| Dataset sizes: 46,186 / 1,939 / 2,382 / 408 / 15,883 | Not present in thesis PDF or active experiment source | **Fail** | Remove under PDF-only scope, or establish a separately authorized data-provenance policy. |
| One gold passage per query | Ch. 4 Evaluation | Pass | Keep. |
| Hit@1 and Hit@20 definitions | Ch. 4 Evaluation | Pass | Keep. |
| SpeechGR 5.81 / 18.452 on test | Table 4.2, Ch. 4 summary, and conclusion; prose says 5.21 | Pass after disclosed typo correction | Keep 5.81/18.452 and retain the ledger note that 5.21 is a thesis prose typo. |
| SpeechDPR with KD: Hit@20 19.73 | Table 4.2, imported comparison | Pass only as prior-work reported | Preserve dagger/source label; do not imply reproduction or matched supervision/split. |
| SpeechDPR without KD: Hit@20 0.04 | Table 4.2, imported comparison | Pass only as prior-work reported | Same as above. |
| SpeechGR is 1.278 Hit@20 points below 19.73 | Correct arithmetic from two reported values | Pass as derived comparison | Keep “points,” not relative percent; continue to state supervision mismatch. |
| BM25 0.0 / 0.0 | Table 4.2 | Numeric value passes; input definition conflicts | Keep only with unresolved-input label, or remove until evaluation script resolves gold text versus ASR text. |
| ASR+BM25 1.5 / 5.2 | Table 4.2 | Numeric value passes; ASR details absent | Keep with protocol caveat; do not infer why it is lower. |
| Main system: Flan-T5-base, layer 22, \(K=500\), 10 pretraining epochs, 6k-hour LibriLight, 15% masking, beam 20 | Ch. 4 Model Configuration | Pass | Keep. Codebook-training data remain internally inconsistent at 100h versus 960h. |
| Final system has no BPE | Inferred from selected \(K=500\), no-BPE ablation row; conflicts with generic Data wording | Ambiguous editorial resolution | Keep only with source-audit note; ideally verify original config before submission. |
| Unit/BPE rows in Table 2 | Table 4.3 | Numeric transcription passes | Keep all values; preserve unresolved split and pretraining-setting caveats. |
| Layer 7 to 22 at \(K=500\): 1.20/6.00 to 3.61/14.189 | Table 4.3 | Arithmetic/transcription passes | Describe as the reported same-\(K\) rows, not a universal layer effect. |
| \(K=128\) best Hit@1 and \(K=500\) best Hit@20 among no-BPE layer-22 rows | Table 4.3 | Pass | Keep, explicitly scoped to those rows. |
| Pretraining 0 / 3 / 10: 3.61/14.189, 4.54/15.73, 5.81/18.452 | Table 4.4 | Numeric transcription passes; row splits unresolved | Keep values; change “improves” to “reported scores increase” and do not call the difference test-set evidence. |
| Pretraining 0-to-10 difference: +2.20 / +4.263 | Correct arithmetic from Table 4.4 | Pass as derived delta | Keep only with the same split caveat. |
| String ID 3.61/14.19; atomic ID 4.54/13.69 | Ch. 4 Identifier Design | Pass | Keep as a metric trade-off, not a statistically established preference. |
| Ranking system approximately 2.73/12.84 | Not printed in active thesis PDF; derived from approximate drops | **Major concern** | Replace with the thesis-reported deltas \(-0.88/-1.349\). |
| Q-Former 3.61/14.189 to 2.73/12.584, \(L=17,Q=1\) | Ch. 4 Q-Former subsection | Pass | Keep; current mechanism caveat is appropriate. |
| VAD 3.61/14.189 to 3.27/12.97 | Ch. 4 VAD subsection | Pass | Keep; present deduplication interaction only as an untested hypothesis. |

## Gate summary

- Thesis-only evidence: **fail** because dataset sizes are outside the thesis
  PDF evidence boundary.
- Protocol honesty: **conditional pass** for split disclosure, but the main
  table needs source/split separation.
- Claim precision: **fail at quality level** because “controlled/improve” and
  some mechanism language exceed the recorded protocol.
- Submission compliance: **pass** for style, anonymity, US Letter, and page
  limit in the current artifact; references begin on content page 6.
- Render quality: **fail** because of two broken references and overlapping
  Figure 1 labels.

Round 2 should be requested only after H1 and M1--M5 are fixed and a fresh PDF
is rendered.
