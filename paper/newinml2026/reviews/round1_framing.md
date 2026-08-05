# Round 1 framing review

## Verdict

**Major revision; not yet at the rubric's “okay” threshold.**

The defensible paper is a feasibility and empirical-design study, and the draft
mostly adopts that level of claim. The central claim is appropriately strong
for thesis-only evidence **if it remains limited to “SpeechGR demonstrates
feasibility on English SLUE-SQA-5.”** It is not strong enough to establish
general superiority, multilingual applicability, or causal design rules, and
the current abstract occasionally crosses that line when it says the ablations
“identify which adaptations matter.”

On page 1, a reviewer can identify the main contribution from the abstract:
discrete-speech inputs are adapted to DocID-generating retrieval and evaluated
on SLUE-SQA-5. The formal contribution bullets begin on page 2, but their
substance is visible in the abstract, so this is acceptable. The **principal
limitation is not visible from page 1**: “supervision, evaluation, and
generalization limits” names categories but does not tell the reviewer that
the evidence is one English benchmark, the SpeechDPR comparison is not
supervision-matched, ablation protocols are partly unresolved, and repeated-run
uncertainty is unreported. Quality gate 8 therefore fails.

## Hard-gate failures

1. **Protocol honesty is contradicted by “controlled” language.** The abstract
   says “controlled experiments identify which adaptations matter,” and the
   third contribution promises “controlled evidence,” while the paper later
   states that ablation splits, pretraining settings, and some shared
   configurations are not unambiguously recorded. These cannot all be
   controlled comparisons on the present evidence. The values themselves are
   thesis-only and disclosed honestly; the failure is the strength of the
   label and inference.

2. **The imported baseline split is over-specified in Table 1.** The caption
   “Main results on the SLUE-SQA-5 test split” applies grammatically to every
   row, but the ledger says the exact prior-work SpeechDPR split has not been
   independently verified. The caption must distinguish the thesis test rows
   from the imported prior-work rows until that audit is complete.

The other strict hard gates appear to pass in the current draft: the
quantitative values are in the ledger; imported SpeechDPR values are marked as
reported; `index-free` is defined as no external embedding index/ANN; the trie
is correctly described as constrained decoding; there is no priority,
state-of-the-art, fully-textless, or demonstrated unwritten-language claim; and
the official anonymous workshop style is used. Content ends before the
eight-page limit and the PDF metadata is anonymous.

## Major concerns, ranked

### 1. The first-page limitation is generic rather than decision-relevant

The abstract is otherwise strong: it states the task, mechanism, headline
values, and positive/negative results in 155 words. Its last sentence,
however, says only that the study makes “supervision, evaluation, and
generalization limits” clear. A skeptical reviewer needs the concrete boundary
immediately: one English dataset, an unmatched imported baseline, unresolved
ablation conditions, and no uncertainty estimates. Without that sentence, the
positive ablation language is easier to read causally than the evidence allows.

### 2. The empirical claims oscillate between feasibility and causal guidance

The main-result language is well calibrated (“evidence that ... is feasible,
not ... state of the art”), and each negative ablation generally follows the
useful hypothesis--result--local-lesson pattern. In contrast, the abstract,
contribution list, pretraining paragraph, and conclusion use “identify,”
“matter,” “improves,” and “beneficial.” Given unresolved splits and shared
settings, the strongest supported wording is that particular reported
configurations are associated with higher or lower scores. Speech-unit
pretraining is the clearest positive pattern, but it is not yet a clean causal
result.

### 3. Novelty is understandable but not yet secured against the 2026 literature

The paper's actual novelty is a clear combination: discrete spoken queries,
spoken passages, and generative DocID retrieval instead of dense-vector search.
The Related Work section establishes each ingredient but does not make that
two-axis contrast explicit enough. It also relies on the thesis bibliography
and does not demonstrate that the positioning remains current in 2025--2026.
The paper wisely avoids a “first” claim, but “Existing GR studies ... have
largely assumed textual queries” still needs an updated literature audit. The
final positioning should contrast (i) non-text objects retrieved from text
queries and (ii) spoken-query dense retrieval with SpeechGR's spoken-query
DocID generation.

### 4. The contribution bullets are partly redundant and undersell the evidence

The first bullet introduces the formulation/system; the second repeats the
same transcript-free/index-free properties without stating what the evaluation
establishes. The second contribution should contain the exact feasibility
result and the supervision-aware status of the SpeechDPR reference. The third
should explicitly say that the paper reports both successful and unsuccessful
thesis ablations while scoping unresolved protocol. This would make the
workshop value much more concrete.

### 5. Audit caveats dominate the experimental narrative instead of being scoped once

The repeated statements about unresolved splits and pretraining appear in the
setup, two table captions, pretraining discussion, and limitations. Honesty is
essential, but repetition makes the paper read like an unfinished artifact
audit. State the evidence boundary once in the setup, keep one short reminder
in each relevant table caption, and then write the analysis in explicitly
configurational language (“among the reported rows”). Use the available page
budget for a sharper task/novelty comparison rather than repeating the same
disclaimer.

### 6. The rendered overview and cross-references fail the presentation gate

The current Figure 1 has overlapping labels around “gold DocID,” the dashed
trie arrow, and “masks invalid next tokens.” More importantly, it visually
connects an inference-time trie to a gold training target, blurring the
paper's central constrained-decoding distinction. The rendered PDF also shows
“Section ??” and “Table ?” because `sec:analysis` is undefined. These are not
framing defects in isolation, but they materially weaken the presentation of
the central claim and fail the no-overlap/no-broken-reference quality gate.

## Minor concerns

- The title's “Exploring” is safe but vague. “A Feasibility Study” would state
  the evidential level more precisely.
- Use “passage identifier” consistently; the abstract opens with “document
  identifiers,” while the task is passage retrieval.
- “Transcript-free” should consistently mean that the benchmark transcripts
  are not consumed by SpeechGR training or inference. It should never imply
  that the Flan-T5 backbone has no text pretraining.
- “GR replaces nearest-neighbor search” is categorical. “In the standard GR
  formulation used here” would avoid excluding hybrid GR systems.
- The conclusion's “near SpeechDPR” is vague. The paper already has the clearer
  statement: 1.278 Hit@20 points lower, under different supervision and
  retrieval mechanisms.
- The Introduction's first paragraph is readable but broad. Its main function
  should be the direct spoken-query retrieval problem, not a general criticism
  of ASR.
- Bibliographic entries rendered from Semantic Scholar contain raw corpus URLs
  and inconsistent venue metadata. This is not a claim problem, but it makes
  the short paper look less publication-ready.

## Exact recommended rewrites

### Abstract

Replace the current abstract, beginning “Generative retrieval maps a query
directly ...,” with:

> Generative retrieval (GR) maps queries to passage identifiers without
> searching an external embedding index, but it has mainly been studied with
> text queries. We present SpeechGR, a feasibility study of GR from spoken
> queries. SpeechGR quantizes spoken queries and passages into discrete HuBERT
> units, adapts a Flan-T5 encoder--decoder, and generates valid passage
> identifiers with trie-constrained decoding; it uses no benchmark query
> transcripts or ANN search. On English SLUE-SQA-5, SpeechGR reaches 5.81\%
> Hit@1 and 18.452\% Hit@20. SpeechDPR reports 19.73\% Hit@20 with knowledge
> distillation, but uses dense retrieval and different supervision, so this is
> contextual rather than a matched comparison. Thesis ablations suggest that
> later-layer acoustic units and longer speech-unit pretraining are associated
> with higher retrieval accuracy, while the tested BPE, Q-Former, ranking-loss,
> and VAD variants are lower than their reported baselines. Evidence is limited
> to one English benchmark, unresolved ablation protocols, and no reported
> repeated-run uncertainty estimates. SpeechGR therefore establishes
> feasibility and a set of design hypotheses, not broad superiority.

This remains within the 150--180-word target and makes the principal limitation
visible on page 1.

### Gap and novelty paragraph

Replace the end of Introduction paragraph 2, beginning “Existing GR studies,
however ...,” after an updated 2025--2026 literature audit:

> Prior non-text GR work in the thesis bibliography generates identifiers for
> non-text objects from textual queries
> \citep{li-etal-2024-generative,Fang2024ACEAG}, whereas SpeechDPR accepts
> spoken queries but searches dense passage representations
> \citep{Lin2024SpeechDPRES}. SpeechGR studies a different design point:
> discrete spoken queries are mapped to spoken-passage DocIDs by a generative
> retriever. Our contribution is this speech-input GR formulation and its
> empirical design study, rather than a new decoding objective.

Do not add a priority claim after the literature audit.

### Contribution bullets

Replace the three bullets after “This work makes three contributions:” with:

> - We formulate spoken-query generative passage retrieval and develop
>   SpeechGR, which maps discrete acoustic units to passage identifiers.
> - On English SLUE-SQA-5, SpeechGR reaches 5.81 Hit@1 and 18.452 Hit@20
>   without benchmark query transcripts or dense ANN search; we contextualize
>   this result against SpeechDPR while keeping supervision and retrieval
>   differences explicit.
> - We report a scoped study of acoustic units, identifier design, speech-unit
>   pretraining, sequence compression, ranking loss, and VAD, retaining both
>   successful and unsuccessful thesis experiments and disclosing unresolved
>   protocol details.

### Evidence-scope statement

Replace the final two sentences of “Data and evaluation,” beginning “It
describes the ablations ...,” with one reusable scope statement:

> The thesis presents the ablations as development-stage experiments but does
> not identify the split or shared pretraining setting for every row. We
> therefore report the values as recorded, leave the split unresolved, and
> interpret them as comparisons among the stated configurations rather than
> as test-set or repeated-run effects.

Then shorten repeated caveats elsewhere instead of restating the full issue.

### Main-table caption

Replace “Main results on the SLUE-SQA-5 test split” with:

> Main comparison as presented in the thesis. SpeechGR, BM25, and ASR+BM25 are
> reported on the thesis test split. SpeechDPR values are imported from prior
> work, not reproduced here; their exact prior-work evaluation split remains
> to be verified, and the knowledge-distilled setting is not
> supervision-matched to SpeechGR.

### Positive-ablation calibration

Replace “Increasing pretraining from 0 to 3 and 10 epochs improves performance
monotonically ... making pretraining the strongest positive intervention” with:

> The reported scores increase across 0, 3, and 10 pretraining epochs, from
> 3.61/14.189 to 4.54/15.73 and 5.81/18.452. This is the strongest positive
> pattern in the thesis, but the unresolved per-row split and shared settings
> prevent a causal or test-set interpretation.

Replace the conclusion phrase “identifies speech-unit pretraining as
beneficial” with:

> reports higher scores for configurations with more speech-unit pretraining
> and lower scores for several tested compression and optimization variants

### Overview figure and references

Redraw Figure 1 with separate training and inference lanes. Training should end
at a **target DocID**; inference should show
**spoken query \(\rightarrow\) Flan-T5 decoder \(\rightarrow\) generated
DocID**, with the valid-DocID trie pointing to the decoder's next-token
constraint rather than to the gold target. Use a short arrow label such as
“valid next tokens” outside the nodes.

Add `\label{sec:analysis}` immediately after
`\subsection{What works and what does not}` so the two `Section
\ref{sec:analysis}` references resolve, then rebuild until the PDF contains no
question-mark references or material overfull boxes.
