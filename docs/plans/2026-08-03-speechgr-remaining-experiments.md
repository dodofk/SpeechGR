# SpeechGR remaining experiment plan

Updated: 2026-08-03

## Decision

The current thesis-derived paper is suitable for NewInML as a feasibility and
design-study submission. It is not ready for ICASSP without a reproducible
canonical checkpoint, matched baselines, and experiments showing that retrieval
depends on question content.

Use 2026-08-20 as the venue gate:

- If the canonical model and the mandatory model diagnostics below are ready,
  decide whether to target ICASSP and avoid an unapproved simultaneous
  submission.
- Otherwise, submit the carefully scoped feasibility paper to NewInML by
  2026-08-29.

Before adding new neural results, explicitly broaden the evidence policy from
“thesis results plus the BM25 reproduction” to “thesis results plus clearly
labelled post-thesis experiments.” Do not mix the two provenances.

## Artifact status and recovery gate

Available:

- Complete 55 MB SLUE-SQA-5 text metadata: 15,883 passages, 2,382 test
  questions, and 408 verified-test questions.
- Public HuBERT layer-22, K=500 packed-unit dataset: about 482 MiB; no raw
  118 GB audio download is needed.
- Joyboy contains a post-thesis QG-augmented retrieval checkpoint and
  validation predictions.

Missing:

- The checkpoint that produced the thesis headline result, 5.81 Hit@1 and
  18.452 Hit@20.
- Per-query test predictions for that model.

Recovery actions:

1. Search the thesis machine, external disks, W&B artifacts, cloud storage, and
   any older model server.
2. Record the checkpoint hash, tokenizer, unit-token lookup, training config,
   and model-selection step if found.
3. Stop recovery-only work after 2026-08-06. If the artifact is still missing,
   either start a newly specified canonical run or keep the workshop claim at
   thesis-level feasibility.

The existing QG-augmented checkpoint must not replace the headline model. Its
negative result becomes interpretable only after an original-query-only run is
trained under the same configuration.

## Work packages

### WP1 — Lexical-overlap and BM25 analysis

Priority: mandatory; no GPU or audio.

Extend the deterministic BM25 evaluation to emit one row per query with:

- question ID and source dataset prefix;
- query length;
- shared-token count and query-token coverage;
- IDF-weighted query-to-gold-passage coverage;
- gold BM25 score and rank;
- Hit@1 and Hit@20.

Report fixed low/middle/high overlap strata, including the bottom quartile,
with 10,000-query bootstrap confidence intervals. The low-overlap set must be
called a naturally occurring low-overlap subset, not a paraphrase subset.

Deliverables:

- per-query CSV/JSON;
- overall and stratified BM25 metrics;
- reusable bootstrap code;
- the data for Figure 2A.

### WP2 — Acoustic-unit sparse retrieval and data diagnostics

Priority: mandatory; CPU, about 263 MB for documents plus test units.

Build unigram and bigram HuBERT-unit TF-IDF/BM25 retrieval baselines over the
same 15,883 passages. Report Hit@1 and Hit@20, plus:

- query and passage unit lengths;
- truncation frequency under the actual model limit;
- exact unit-sequence collisions;
- repetition-collapse ratio;
- performance by query-length and lexical-overlap stratum.

Decision criterion: if simple unit similarity approaches SpeechGR, the paper
must weaken the learned question-to-passage mapping claim. If it remains near
random while SpeechGR is clearly higher, it supports the value of learned
retrieval rather than acoustic matching.

Deliverables:

- deterministic unit-baseline artifact;
- collision/truncation report;
- unit-baseline series for Figure 2A and Table 1.

### WP3 — Evaluation-only SpeechGR runner

Priority: mandatory if a canonical checkpoint is available.

Add an evaluation command that does not call training. It must:

- load a specified checkpoint and unit-token lookup;
- select validation, test, or verified-test;
- run beam-20 trie-constrained decoding;
- save question ID, gold DocID, ordered top-20 DocIDs, and scores;
- verify that every output is a valid corpus identifier;
- preserve the candidate-pool manifest and checkpoint hash.

Expected test inference time on Joyboy is about two minutes after the correct
checkpoint is available.

### WP4 — Question-content sanity controls

Priority: highest-value model experiment; depends on WP3.

Evaluate the fixed canonical model with:

1. the correct question units;
2. a fixed permutation assigning each example another question;
3. within-question unit-order shuffling with three fixed seeds;
4. unit deletion or masking at 5%, 10%, and 20%;
5. an empty/minimal query input if the model interface supports it safely.

Report Hit@1/20 and paired bootstrap differences against the correct-query
condition.

Decision criterion: correct questions should substantially exceed wrong-query,
shuffled, and empty-input controls. Otherwise the paper cannot claim strong
question-content sensitivity.

### WP5 — SpeechGR lexical stratification and verified test

Priority: mandatory for the stronger paper; depends on WP1 and WP3.

Join the canonical per-query predictions to the overlap bins frozen in WP1.
Report SpeechGR, BM25, unit sparse retrieval, and random retrieval in the same
bins. Evaluate the 408-query verified-test split separately and provide query
bootstrap intervals.

Decision criterion: above-random low-overlap performance supports a bounded
content-sensitive retrieval claim. Collapse in the lowest-overlap bin supports
only a lexical-association feasibility claim.

### WP6 — Matched neural baselines

Priority: required for ICASSP; optional for NewInML.

In decreasing priority:

1. canonical original-query-only SpeechGR under a fully recorded config;
2. the same system without speech-unit pretraining;
3. gold-text GR using the same Flan-T5 size, DocIDs, training pairs, split, and
   candidate collection;
4. three seeds for the canonical SpeechGR configuration.

Observed runtime suggests about 45 GPU-hours per 100k-step retrieval run, or
about 135 GPU-hours for three sequential seeds. Do not rerun the full thesis
architecture grid.

The existing QG-augmentation result may be reported as a negative experiment
only after comparison with item 1 under matched conditions.

### WP7 — Candidate-pool sensitivity

Priority: useful supporting analysis.

Evaluate fixed global pools containing all 1,969 unique test gold passages at
1,969, 5,000, 10,000, and 15,883 candidates. Add non-gold passages using five
fixed seeds and save every pool manifest. Run BM25 first; extend to SpeechGR
only if the evaluator supports the corresponding valid-DocID tries.

Do not describe this as matching SpeechDPR's roughly 39k pool because those
additional candidates are unavailable.

## Experiments not required before submission

- More VAD, Q-Former, ranking-loss, BPE, or broad codebook-size ablations.
- Full 118 GB audio download.
- Waveform noise/reverb experiments before the unit and content controls are
  complete.
- Additional unfiltered pseudo-query generation.

## Paper deliverables

Keep:

- Figure 1: SpeechGR architecture and two training paths.
- Table 1: matched core results and clearly separated prior-work context.

Add Figure 2 with two panels:

- A: Hit@20 across lexical-overlap strata for SpeechGR, BM25, unit sparse, and
  random retrieval.
- B: correct-query versus wrong/shuffled/corrupted-query controls.

Replace the two current ablation tables with one compact table containing only
the selected layer/K setting, with/without speech-unit pretraining, and at most
one matched BPE comparison. Move identifier, ranking-loss, Q-Former, and QG
negative results to a short paragraph or supplementary material if allowed.

## Schedule

- Aug 3–6: evidence-policy decision, checkpoint recovery, conference-first
  citation cleanup.
- Aug 7–10: WP1 and WP2.
- Aug 10–14: WP3; begin a new canonical run if recovery failed and new neural
  evidence is authorized.
- Aug 14–20: WP4 and WP5, or finish the canonical run.
- Aug 20: venue/readiness gate.
- Aug 21–24: WP6/WP7 only where they strengthen a matched claim.
- Aug 24–27: rewrite figures, tables, abstract, and results.
- Aug 27–28: literature, evidence, and rendered-layout review.
- Aug 29: NewInML submission if that path is selected.

For an ICASSP path, freeze experiments no later than Sep 3, compress to four
technical pages plus one references-only page, and reserve Sep 10–15 for
coauthor and final compliance review before the Sep 16 deadline.
