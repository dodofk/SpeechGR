# SpeechGR to NewInML 2026 conversion plan

Date: 2026-07-31

## Decision

Submit SpeechGR to NewInML at NeurIPS 2026, provided the author satisfies the
workshop's newcomer eligibility rule. The workshop is a strong fit because it:

- welcomes all machine-learning topics;
- accepts 2--8 content pages, excluding references;
- uses double-blind review;
- is non-archival; and
- is explicitly designed to give first-time top-venue authors research and
  paper feedback.

Official call: <https://newinml.github.io/NewInML2026NeurIPS/>

Listed deadline: 2026-08-29, 23:59 AoE. The page also says that dates are being
updated, so check the call and OpenReview again before submission.

## Recommended paper

Working title:

> SpeechGR: Exploring Generative Retrieval Directly from Spoken Queries

Recommended central claim:

> SpeechGR shows that a generative retriever can map discrete acoustic units
> from spoken queries directly to passage identifiers without query
> transcripts or a dense retrieval index. On SLUE-SQA-5, it approaches
> the reported Hit@20 of a knowledge-distilled speech dense retriever. A
> systematic study of unit extraction, identifier design, speech-unit
> pretraining, ranking loss, sequence compression, and VAD shows both what
> helps and what fails in this new setting.

This is a feasibility and empirical-design paper, not a state-of-the-art paper.
Negative results are part of the contribution when they test a clear
hypothesis under a controlled configuration.

## Claims to use and avoid

### Use

- "transcript-free at SpeechGR training and inference time," if confirmed by
  the final data/code audit;
- "index-free," using the standard generative-retrieval meaning: no external
  document embedding index or ANN search;
- "generates passage identifiers from spoken queries";
- "approaches SpeechDPR's reported Hit@20 under different supervision";
- "a feasibility study of speech-input generative retrieval";
- "English SLUE-SQA-5 experiments."

### Avoid or qualify

- **"fully textless":** Flan-T5 is text-pretrained, the identifier space is
  textual/metadata-derived, and the benchmark comes from English Wikipedia.
- **Undefined "index-free":** define it once as the absence of an external
  document embedding index or ANN search. A trie over valid DocID prefixes is
  a standard constrained-decoding mechanism in generative retrieval, not a
  dense retrieval index, and does not conflict with this claim.
- **"for unwritten languages":** this is motivation or future work, not a
  demonstrated result.
- **"state of the art":** SpeechGR does not beat SpeechDPR and the baseline
  suite is incomplete.
- **"first":** use only after an updated 2025--2026 literature search and
  phrase the comparison precisely (spoken-query generative passage retrieval).
- **directly comparable supervision:** SpeechDPR's 19.73 Hit@20 uses knowledge
  distillation, while its reported no-KD result is 0.04. Put supervision and
  index type in the main table.

## Thesis-only evidence policy

The workshop paper reports only experiments already present in the completed
master's thesis PDF. Do not add results from the current QG work, SimulGR, new
checkpoints, or new baselines. We may reorganize the experiments, correct
obvious writing/arithmetic errors, and improve their interpretation, but we do
not silently replace the thesis evidence.

Before drafting, create a result ledger containing the thesis page/table,
configuration, split as reported, Hit@1, Hit@20, and the paper section where the
result will appear.

1. **Resolve internal inconsistencies editorially**
   - Main-results prose says Hit@1 5.21; the table, abstract, and conclusion say
     5.81. Use 5.81 as the intended value and record 5.21 as a corrected typo.
   - The abstract says SpeechGR is within one percentage point of SpeechDPR at
     Top-20; 19.73 - 18.452 = 1.278 points.
   - The thesis says ablations use development data, but the 10-epoch
     pretraining row exactly matches the headline test result. Label every row
     using only what the thesis establishes; do not invent a split.
   - The method says speech pretraining internalizes the corpus index, although
     the described LibriLight span-corruption stage adapts the acoustic-token
     model and does not index SLUE passages.

2. **Preserve protocol clarity**
   - Report 46,186 train queries, 1,939 validation queries, 2,382 test queries,
     408 verified-test queries, and 15,883 indexed passages, after confirming
     that these counts describe the same thesis dataset.
   - Explain how passage-indexing examples and spoken-query retrieval examples
     are mixed during training.
   - State the identifier construction, maximum input length/truncation,
     constrained beam implementation, beam size, training steps/epochs,
     optimizer, learning rate, batch size, model-selection split, and hardware.
   - Clarify whether BPE is absent from the final K=500 system.

3. **Use thesis baselines honestly**
   - Separate ground-truth-text BM25 from ASR+BM25; the thesis currently
     describes BM25 both ways.
   - Verify all imported SpeechDPR numbers and their evaluation split.
   - Put modality, transcript use, knowledge distillation, and external index
     type beside each result.
   - Clearly mark SpeechDPR numbers as reported from prior work rather than
     reproduced in the thesis.

4. **Do not manufacture completeness**
   - If the thesis does not report seeds, confidence intervals, or a necessary
     implementation detail, say so in limitations.
   - Do not calculate an unreported experimental result from current
     checkpoints.

## The 8-page strategy

Do not summarize every thesis chapter. Rebuild the paper around one argument:

1. Existing speech retrieval either transcribes speech or uses dense retrieval.
2. Generative retrieval offers a transcript-free, index-free alternative, but
   has been developed mainly for textual queries.
3. SpeechGR makes generative retrieval accept spoken queries through discrete
   acoustic units.
4. SLUE-SQA-5 results establish feasibility, and a broad set of successful and
   unsuccessful ablations explains which parts of the recipe matter.

Everything that does not support this chain is expendable.

Target **7.6 content pages**, leaving approximately 0.4 page for float movement
and final edits. References start after page 8.

| Part | Budget | Required content |
|---|---:|---|
| Title and abstract | 0.45 | One 150--180 word abstract; no broad future-work claims |
| 1. Introduction | 0.85 | Problem, gap, SpeechGR, headline evidence, 3 contributions |
| 2. Related work | 0.45 | GR, speech retrieval, and discrete speech units only |
| 3. SpeechGR | 1.65 | Task, units, model/pretraining, indexing/retrieval objective, constrained decoding, one figure |
| 4. Experimental setup | 0.75 | Data, baselines, metrics, splits, and thesis configuration |
| 5. Main results | 0.75 | Main comparison table plus supervision-aware interpretation |
| 6. What works and what does not | 2.10 | Unit, ID, pretraining, ranking, Q-Former, BPE, and VAD findings |
| 7. Limitations and conclusion | 0.45 | Honest scope and a short closing paragraph |
| **Total** | **7.45** | 0.55-page safety margin |

### Introduction: four paragraphs only

1. **Problem:** retrieving spoken passages directly from spoken queries matters
   when transcripts are unavailable or undesirable.
2. **Gap:** speech retrieval is dominated by ASR pipelines or dense indices;
   GR removes the dense index but has mostly assumed text queries.
3. **Approach and result:** introduce SpeechGR in 4--5 sentences and give the
   verified headline numbers.
4. **Contributions:** exactly three bullets:
   - speech-input GR formulation and system;
   - transcript-free, index-free evaluation on SLUE-SQA-5;
   - controlled evidence about successful and unsuccessful design choices.

### Method: what deserves space

- Define the retrieval task and DocID generation in one short paragraph.
- Give the HuBERT/k-means quantization equation and explain deduplication.
- Explain identifier construction in one paragraph.
- Explain Flan-T5 vocabulary expansion and LibriLight span-corruption
  pretraining.
- State the indexing and query-to-DocID training examples clearly.
- Give the cross-entropy objective.
- Explain trie-constrained beam search in 3--4 sentences.
- Use one simplified pipeline figure occupying at most half a page.

The detailed Q-Former and ranking-loss mathematics belong in neither the main
method nor the introduction; they are failed ablations and need only a short
description in the analysis.

### Results and empirical-study budget

Use at most **one figure and three compact tables** in the entire paper.

1. **Figure 1: SpeechGR overview**
   - waveform to HuBERT units to Flan-T5 to DocID;
   - show passage indexing and spoken-query retrieval as two training paths;
   - show the DocID trie only at inference.

2. **Table 1: Main comparison**
   - columns: model, query input, transcript supervision, retrieval mechanism,
     Hit@1, Hit@20;
   - rows: verified BM25 variants, SpeechDPR without KD, SpeechDPR with KD,
     SpeechGR;
   - label imported versus reproduced results.

3. **Table 2: Core design analysis**
   - panel A: all complete HuBERT layer/codebook/BPE configurations reported
     in the thesis;
   - panel B: 0, 3, and 10 epochs of speech-unit pretraining;
   - label the evaluation split in the caption.

4. **Table 3: Secondary ablations**
   - identifier type, ranking loss, Q-Former, and VAD as compact grouped rows;
   - show the relevant baseline beside each intervention so a negative result
     is interpretable;
   - include only experiments with complete numbers and a sufficiently clear
     comparison in the thesis.

Each negative experiment should follow the same three-sentence pattern:

1. hypothesis: why the intervention might help;
2. result: the exact change in Hit@1/Hit@20;
3. lesson: what the failure suggests, without claiming a universal conclusion.

This lets the paper report more of the work without spending method-sized space
on every failed intervention.

### Thesis experiments selected for the paper

**Keep as headline evidence**

- Main SpeechGR result: 5.81 Hit@1 and 18.452 Hit@20.
- Reported SpeechDPR comparison: 19.73 Hit@20 with KD and 0.04 without KD.
- BM25/ASR+BM25 only after their input definitions are made consistent.

**Keep as core findings**

- HuBERT layer 7 versus layer 22.
- Codebook sizes 128, 500, 1000, and 2000.
- BPE variants, because the consistent degradation is a useful negative
  finding about compressing acoustic-token sequences.
- Speech-unit pretraining at 0, 3, and 10 epochs, the clearest positive
  ablation in the thesis.
- String versus atomic identifiers, presented as a Hit@1/Hit@20 trade-off
  rather than a simple winner.

**Keep as compact negative findings**

- Pairwise ranking loss.
- Window-level Q-Former compression.
- Voice-activity detection.

**Omit**

- The 128-token input-length experiment: the thesis discusses it but does not
  provide a complete active results table.
- Backbone-size claims without a complete reported comparison table.
- Any experiment found only in current repository logs or post-thesis work.
- Any unfinished/commented table with missing values.

### Keep/cut map from the thesis

| Thesis material | Workshop action |
|---|---|
| English abstract | Rewrite from zero after results are frozen |
| Introduction, general IR history | Cut |
| Text availability/unwritten-language motivation | Reduce to 2--3 cautious sentences |
| GR gap and SpeechGR overview | Keep and sharpen |
| Three thesis objectives | Convert into three contribution bullets |
| Literature-review chapter | Cut approximately 80--90% |
| GR identifier taxonomy | Compress to one related-work paragraph |
| General data-augmentation/ranking tutorial | Cut |
| Speech retrieval and discrete-unit work | Keep as two compact paragraphs |
| SpeechGR pipeline | Keep, but redraw for paper readability |
| HuBERT quantization and deduplication | Keep |
| Q-Former derivation | Cut; report only as a negative ablation |
| Identifier design | Keep one method paragraph and one result row |
| Flan-T5 adaptation and pretraining | Keep |
| Training objective | Keep only the main CE equation |
| Ranking-loss derivation | Cut; cite and summarize in one sentence |
| Trie-constrained decoding | Keep |
| Dataset and model configuration | Keep, rewrite as a reproducible setup |
| Main results | Keep after thesis-table consistency checks |
| Unit and pretraining ablations | Keep as the main analysis |
| BPE, Q-Former, ranking loss, and VAD | Compress into one small table/paragraph |
| Speculative result explanations | Keep only explanations supported by evidence |
| Future-work chapter | Cut; retain at most two sentences in limitations |
| Thesis conclusion | Rewrite as a 5--6 sentence paper conclusion |

### Material that should not consume the 8 pages

- thesis administration, acknowledgments, Chinese abstract, chapter summaries;
- broad textbook explanations of BM25, dense retrieval, T5, or HuBERT;
- full surveys of every DocID family;
- derivations for methods that did not improve the final model;
- repeated statements that the system is transcript-free and index-free;
- unsupported multilingual or unwritten-language impact claims;
- long future-work lists.

Do not plan around a free appendix: the workshop call explicitly excludes
references from the 8-page limit but does not say that appendices are excluded.
Supplementary material can hold implementation details only if OpenReview
allows it, and the main paper must remain self-contained.

## Drafting order

Write the paper in evidence order, not reading order.

1. **Freeze a result ledger**
   - one row per reported number with thesis page/table, split as reported,
     configuration, metric, and whether it is a thesis result or imported
     comparison;
   - resolve all metric inconsistencies before prose is copied.
2. **Create the official paper shell**
   - NeurIPS 2026 `dblblindworkshop` format, US Letter, anonymous authors;
   - add section headings, empty figure/table slots, and the bibliography first.
3. **Build tables and Figure 1**
   - tables determine the actual claims and prevent the prose from drifting;
   - redraw the pipeline rather than shrinking the thesis figure.
4. **Write setup and results**
   - these sections are the factual core and should be reviewed first.
5. **Write the method**
   - include only the components necessary to reproduce and interpret the
     reported system.
6. **Write related work and introduction**
   - position the now-fixed evidence; do not promise more than the tables show.
7. **Write limitations, conclusion, and abstract last**
   - the abstract is a compressed report of the final paper, not a planning
     statement.
8. **Render and trim**
   - first target: 7.5--7.8 pages;
   - cut repetition, literature detail, and negative-ablation explanation in
     that order;
   - never change template margins or fonts to recover space.

## Scope freeze

This paper is the workshop version of the completed SpeechGR thesis. Do not
merge SimulGR, streaming retrieval, the newer QG work, or a new model
architecture into it. Those directions need a separate paper and would make
the 8-page story incoherent.

The experimental boundary is the master's thesis PDF. The extra experiments
the paper discusses are the already completed negative ablations from that
thesis, not new experiments to be launched for this deadline.

### Required submission

- a thesis-result ledger and corrected headline evaluation;
- corrected splits, counts, configurations, and metric inconsistencies;
- a supervision-aware table using the baselines reported in the thesis;
- the unit, identifier, pretraining, ranking, Q-Former, BPE, and VAD analyses
  selected above;
- anonymized 8-page paper in the official template.

The complete first PDF should be ready by **August 10**.

### Permitted strengthening

- update the 2025--2026 related-work coverage;
- improve the pipeline figure and table design;
- clarify the original experimental protocol from the thesis source;
- correct typos and arithmetic;
- improve interpretation while keeping conclusions local to the experiment.

Do not add a new result row after the result ledger is frozen. A complete,
internally consistent conversion is the goal.

## Proposed abstract after evidence verification

Generative retrieval replaces embedding search with direct generation of
document identifiers, but prior work primarily assumes textual queries. We
present SpeechGR, a generative retriever that accepts spoken queries represented
as discrete HuBERT units and generates passage identifiers with a
speech-adapted encoder--decoder model. SpeechGR requires neither query
transcripts nor an external document embedding index or approximate-nearest-
neighbor search. Following standard generative retrieval, a trie constrains
decoding to valid identifiers. On SLUE-SQA-5, SpeechGR reaches
5.81 Hit@1 and 18.452 Hit@20, approaching the 19.73 Hit@20 reported for a
knowledge-distilled speech dense retriever under a different supervision
setting. Controlled ablations show that higher HuBERT layers, a mid-sized
acoustic codebook, and speech-unit span-corruption pretraining are important,
whereas BPE compression, a windowed Q-Former, pairwise ranking loss, and
voice-activity detection do not improve retrieval in our setting. These results
establish speech-input generative retrieval as a viable direction while
highlighting the evaluation and scaling challenges that remain.

The numbers in this abstract are placeholders until the evidence gate passes.

## Schedule

- **July 31--August 2:** extract all selected thesis results into the result
  ledger and resolve internal inconsistencies.
- **August 3:** freeze the central claim, table contents, and paper outline.
- **August 4--6:** create the official LaTeX shell, Figure 1, and all tables.
- **August 7--8:** write experimental setup, results, and analysis.
- **August 9--10:** write method, introduction, related work, limitations,
  conclusion, and abstract; render the first complete PDF.
- **August 11--17:** update related work, refine the figure/tables, and conduct
  technical review without adding new experimental results.
- **August 18--21:** advisor/coauthor review and claim/protocol audit.
- **August 22--25:** revise, render, and perform page-limit, reference,
  reproducibility, and anonymization checks.
- **August 26--27:** prepare OpenReview metadata and any permitted
  supplementary material.
- **August 28:** submit one day early.

## Submission checklist

- Use `\usepackage[dblblindworkshop]{neurips_2026}` and set
  `\workshoptitle{New In ML at NeurIPS 2026}`.
- Use US Letter, not the A4 thesis layout.
- Keep content to at most 8 pages, excluding references.
- Remove author names, affiliation, acknowledgments, thesis DOI, repository
  links tied to the author, and identifying self-references from the review
  version.
- Confirm workshop eligibility for every author.
- Because the master's thesis may be public, ask the organizers whether its
  public DOI requires any special disclosure under their double-blind policy.
- Recheck the official call and OpenReview deadline immediately before
  submission.
