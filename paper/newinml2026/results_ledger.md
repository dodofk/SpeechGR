# SpeechGR evidence and result ledger

Evidence boundary: the completed master's thesis, one explicitly labelled
workshop BM25 reproduction, and published prior-work context. No post-thesis
neural-model result is used. Thesis values below are copied from
`contents/experiment.tex` and cross-checked against the thesis abstract and
conclusion where noted. A row marked “prior-work reported” is not a
reproduction by the thesis author.

Dataset counts used in the revised paper are cross-checked against the official
SLUE Phase-2 paper and the local BM25 inputs.

| Thesis source | Result provenance | Evaluation split | Configuration | Hit@1 | Hit@20 | Intended paper location | Caveat |
|---|---|---|---|---:|---:|---|---|
| Ch. 4, Main Results, Table `tab:main-results` | thesis experiment | test (explicit in table caption) | BM25 | 0.0 | 0.0 | Excluded from revised table | The chapter prose calls this ground-truth-text BM25, while the baseline table describes BM25 over ASR transcripts. The new reproduction obtains 12.72/43.20; no original BM25 evaluator or artifact survives, so the thesis row is treated as an unresolved evaluation/configuration error. |
| Ch. 4, Main Results, Table `tab:main-results` | thesis experiment | test (explicit in table caption) | ASR + BM25 | 1.5 | 5.2 | Excluded from revised table | ASR outputs, model, preprocessing, and retrieval protocol do not survive. The only ASR script does not execute BM25 and references a nonexistent dataset field. |
| SpeechDPR, ICASSP 2024, Table 1 | prior-work reported | SLUE-SQA-5 test | SpeechDPR with knowledge distillation and external dense passage search | — | 19.73 | Table 1, prior-work context | Imported, not reproduced. SpeechDPR searches roughly 39k Spoken Wikipedia passages (427 hours), versus SpeechGR's 15,883 linked-passage collection. Supervision and retrieval mechanism also differ. |
| SpeechDPR, ICASSP 2024, Table 1 | prior-work reported | SLUE-SQA-5 test | SpeechDPR without teacher knowledge distillation and with external dense passage search | — | 0.04 | Table 1, prior-work context | Imported, not reproduced. The speech encoder still uses text-pretrained RoBERTa sentence encoders. Same roughly 39k candidate archive. |
| Ch. 4, Main Results, Table `tab:main-results`; thesis abstract and conclusion | thesis experiment | test (explicit in table caption) | SpeechGR; Flan-T5-base; HuBERT-large layer 22; \(K=500\); no BPE; 10 epochs of LibriLight span-corruption pretraining; constrained beam size 20 | 5.81 | 18.452 | Table 1 and headline result | The Main Results prose says 5.21, but the table, abstract, quantitative summary, and conclusion say 5.81; treat 5.21 as a typo. The thesis abstract's “within one percentage point” claim is also incorrect: \(19.73-18.452=1.278\) points. |
| Ch. 4, Ablations, Table `tab:granularity` | thesis experiment | not unambiguously stated | HuBERT-large layer 7, \(K=500\), no BPE | 1.2 | 6 | Table 2A, acoustic units | The thesis says ablations initially use development data, but does not label the split per row. Some granularity experiments omit speech pretraining, without identifying which rows. |
| Ch. 4, Ablations, Table `tab:granularity` | thesis experiment | not unambiguously stated | HuBERT-large layer 22, \(K=128\), no BPE | 3.76 | 13.04 | Table 2A, acoustic units | Same split/pretraining ambiguity as above. |
| Ch. 4, Ablations, Table `tab:granularity` | thesis experiment | not unambiguously stated | HuBERT-large layer 22, \(K=500\), no BPE | 3.61 | 14.189 | Table 2A and Table 3 baseline | Same split/pretraining ambiguity as above. This is the recurring ablation baseline. |
| Ch. 4, Ablations, Table `tab:granularity` | thesis experiment | not unambiguously stated | HuBERT-large layer 22, \(K=1000\), no BPE | 3.44 | 12.72 | Table 2A, acoustic units | Same split/pretraining ambiguity as above. |
| Ch. 4, Ablations, Table `tab:granularity` | thesis experiment | not unambiguously stated | HuBERT-large layer 22, \(K=500\), BPE vocabulary 1000 | 1.85 | 9.78 | Table 2A, acoustic units | Same split/pretraining ambiguity as above. Directly matched to the \(K=500\), no-BPE row only if the unresolved pretraining setting is also the same. |
| Ch. 4, Ablations, Table `tab:granularity` | thesis experiment | not unambiguously stated | HuBERT-large layer 22, \(K=1000\), BPE vocabulary 2000 | 1.97 | 9.11 | Table 2A, acoustic units | Same split/pretraining ambiguity as above. Directly matched to the \(K=1000\), no-BPE row only if the unresolved pretraining setting is also the same. |
| Ch. 4, Ablations, Table `tab:granularity` | thesis experiment | not unambiguously stated | HuBERT-large layer 22, \(K=2000\), BPE vocabulary 6000 | 1.47 | 7.68 | Table 2A, acoustic units | There is no reported \(K=2000\), no-BPE control, so this row cannot isolate the effect of BPE. |
| Ch. 4, Ablations, Identifier Design subsection | thesis experiment | not unambiguously stated | string/multi-token corpus identifier | 3.61 | 14.19 | Table 3, identifier design | The subsection says “default configuration” but does not identify the split. The value is reported as 14.19 here versus 14.189 elsewhere; preserve the subsection's precision. |
| Ch. 4, Ablations, Identifier Design subsection | thesis experiment | not unambiguously stated | atomic/unstructured identifier | 4.54 | 13.69 | Table 3, identifier design | The identifier construction is described, but the split, tokenizer realization, and exact shared training configuration are not unambiguously reported. |
| Ch. 4, Ablations, Table `tab:pretrain` | thesis experiment | not unambiguously stated | 0 speech-unit pretraining epochs | 3.61 | 14.189 | Table 2B, speech-unit pretraining | The thesis says ablations initially use development data, but does not label this row's split. |
| Ch. 4, Ablations, Table `tab:pretrain` | thesis experiment | not unambiguously stated | 3 epochs on the 6k-hour LibriLight subset; 15% span corruption | 4.54 | 15.73 | Table 2B, speech-unit pretraining | Same split ambiguity as above. |
| Ch. 4, Ablations, Table `tab:pretrain` | thesis experiment | not unambiguously stated | 10 epochs on the 6k-hour LibriLight subset; 15% span corruption | 5.81 | 18.452 | Table 2B, speech-unit pretraining | This exactly matches the headline test result although the surrounding section presents the study as an ablation. Do not assign a split without further evidence. |
| Ch. 4, Ablations, Ranking Loss subsection | thesis experiment | not unambiguously stated | cross-entropy baseline; HuBERT layer 22, \(K=500\) | 3.61 | 14.189 | Table 3, ranking loss | The split and pretraining setting are not unambiguously reported. |
| Ch. 4, Ablations, Ranking Loss subsection | thesis experiment | not unambiguously stated | cross-entropy plus pairwise margin ranking loss with in-batch negatives | reported \(\Delta\approx-0.88\) | reported \(\Delta\approx-1.349\) | Table 3, ranking loss | The PDF reports approximate drops from the 3.61/14.189 baseline, not absolute scores. Preserve the deltas rather than reconstructing result values. |
| Ch. 4, Ablations, Window-level Q-Former subsection | thesis experiment | not unambiguously stated | no Q-Former baseline; HuBERT layer 22, \(K=500\) | 3.61 | 14.189 | Table 3, sequence compression | The split and pretraining setting are not unambiguously reported. |
| Ch. 4, Ablations, Window-level Q-Former subsection | thesis experiment | not unambiguously stated | window Q-Former, window length \(L=17\), one query \(Q=1\) | 2.73 | 12.584 | Table 3, sequence compression | Controlled comparison is stated, but the split and pretraining setting are not. The thesis's information-loss explanation is a hypothesis, not a measured mechanism. |
| Ch. 4, Ablations, VAD subsection | thesis experiment | not unambiguously stated | raw-audio/no-VAD baseline; HuBERT layer 22, \(K=500\) | 3.61 | 14.189 | Omitted from workshop narrative | The split and pretraining setting are not unambiguously reported. |
| Ch. 4, Ablations, VAD subsection | thesis experiment | not unambiguously stated | Silero VAD with its default speech-probability threshold, applied to queries and passages | 3.27 | 12.97 | Omitted from workshop narrative | The split and pretraining setting are not unambiguously reported. The deduplication explanation is plausible but not directly tested. |

## New workshop BM25 reproduction

| Provenance | Split / candidates | Configuration | Hit@1 | Hit@20 | Additional checks |
|---|---|---|---:|---:|---|
| `scripts/evaluate_bm25.py`; `output/metrics/bm25_slue_sqa5_text.json` | test; 2,382 queries against 15,883 unique linked passages | `rank-bm25==0.2.2` `BM25Okapi`; \(k_1=1.5\), \(b=0.75\), \(\epsilon=0.25\); `normalized_question_text`; normalized corpus `document_text`; lowercase `[a-z0-9]+` tokens; stable corpus-row tie-break | 12.7204 | 43.1990 | 100% gold-ID coverage; zero zero-score queries |
| same reproduction | verified-test; 408 queries against the same 15,883 passages | same | 36.0294 | 79.4118 | 100% gold-ID coverage; zero zero-score queries; retained in ledger rather than main table |

Input hashes:

- corpus CSV SHA-256:
  `86c27226d347f0403a75097d2db275311a0ba044b079c52f194139a901775dac`
- test CSV SHA-256:
  `d79d2f11803768b06f3b32f0db6942b1e8300ec64b9cfe1e8a6b3711a06d7677`
- verified-test CSV SHA-256:
  `cef60f3f43bb4105bfb631fc54af5beefef707df71f38c5775cbd538c65956a7`

## Contemporary prior-work context

| Source | Evaluation | Configuration | R@1 | R@10 | Caveat |
|---|---|---|---:|---:|---|
| CLSR, AAAI 2026, Table 2 | SLUE-SQA-5 test, question-to-context | transcript-supervised ASR losses; speech-to-text-like VQ bridge; frozen BGE-base text encoder | 30.65 | 74.43 | Imported, not reproduced; candidate-pool construction is not reported clearly enough to match SpeechGR. |
| CLSR, AAAI 2026, Table 2 | SLUE-SQA-5 test, question-to-context | clean-text BGE reference | 38.71 | 84.34 | Imported, not reproduced; gold text and under-specified candidate pool. |
| WavRAG, ACL 2025, Table 1 | SLUE-SQA-5 speech-to-speech test | Qwen2-Audio retriever; roughly 1.5M mixed-modality examples including text-retrieval and TTS-derived data | 33.92 | 72.21 | Imported, not reproduced; candidate-pool construction is under-specified. |

## Modality-gap literature boundary

- GRACE (ACL 2024), CART (ACL 2025), and GTA (Interspeech 2025) generate
  identifiers for image, audio, or video targets from natural-language queries.
- IRGen (ECCV 2024) is a counterexample to the broad text-query-only claim: it
  maps a raw image to a valid learned identifier for a relevant database image,
  with shared dataset class labels defining relevance in its reported tasks.
- GENIUS (CVPR 2025) accepts text, image, and image--text query content, but
  pairs every query with a natural-language task instruction and uses
  CLIP-based, contrastively aligned encoders.
- The workshop claim is therefore deliberately speech-specific: direct
  spoken-question autoregressive passage-DocID generation remains largely
  unexplored. The manuscript treats IRGen as the closest structural analogue
  among the systems considered, not as a numerical comparator, and does not
  claim to be the first.

## Editorial decisions frozen for the workshop draft

- Use **5.81/18.452** as SpeechGR's headline test result. Record **5.21** as a
  corrected prose typo, not as a second run.
- Do not describe SpeechGR as “within one point” of SpeechDPR. Although the
  arithmetic difference is 1.278, SpeechDPR uses roughly 39k candidates versus
  SpeechGR's 15,883, so the direct gap is not protocol-valid.
- The stated 0-to-10-epoch contrast, \(+2.20\) Hit@1 and \(+4.263\) Hit@20,
  is transparently derived from the two reported pretraining rows
  (\(5.81-3.61\) and \(18.452-14.189\)); it is not presented as a new run.
- Mark both SpeechDPR rows as imported prior-work results and avoid claiming
  matched supervision.
- Treat the new BM25 result as a workshop reproduction in the ledger and a
  gold-text evaluation reference in the manuscript; never blend it with the
  thesis evidence. Treat the thesis's 0.00/0.00 as not reproduced.
- Define the task relation as directed question-to-answer-containing-passage
  relevance rather than acoustic similarity. Because SLUE-SQA-5 is
  text-derived, do not claim that the results isolate semantic reasoning from
  lexical correspondence.
- Do not compute direct gaps between SpeechGR and SpeechDPR/CLSR/WavRAG because
  the candidate pools are different or under-specified.
- Use “not unambiguously stated” for ablation splits until the original
  evaluation artifacts establish otherwise.
- Omit the incomplete maximum-input-length experiment and the unreported
  backbone-size comparison.
- The final \(K=500\) system is shown without BPE, consistent with the selected
  granularity row; the thesis Data section's generic statement that all speech
  was processed with BPE should be corrected in the workshop paper.

## Protocol items still requiring source audit

- Reconstruct ASR+BM25 only if the ASR outputs, model, document/query
  preprocessing, and ID mapping can be recovered under a defined protocol.
- Resolve the thesis's LibriSpeech codebook-training discrepancy (100 hours in
  Model Configuration versus 960 hours in the granularity subsection).
- Recover maximum input length/truncation, optimizer, learning rate, batch
  size, fine-tuning steps/epochs, model-selection rule, hardware, random seeds,
  and number of runs from the original thesis configs. None is unambiguously
  reported in the active thesis text.
