# SimulGR: Simultaneous Generative Retrieval for Spoken Queries

**Research proposal — v0.3, 2026-07-14 (design complete: decisions locked, backbone precedent surveyed, data protocol + TTS budget verified, SpokenQG scoped)**
**Author: Yu-Hsiang (Ricky) Liu**

One-line thesis: a decoder-only, speech-native generative retriever that decodes
semantic document identifiers *while the user is still speaking*, committing
coarse-to-fine as evidence accrues — the retrieval interface duplex voice
agents actually need.

---

## 1. Problem and positioning

Voice agents retrieve in one of two ways today:

- **Turn-final cascade** (production norm, incl. Google S2R's deployment
  context and VoiceAgentRAG): wait for end-of-utterance → ASR/embed → search.
  Retrieval latency lands entirely inside the user-perceived response gap.
- **Between-turn caching** (VoiceAgentRAG, Salesforce 2026): prefetch predicted
  follow-up topics during 3–7s inter-turn silences. Works only in strictly
  turn-based dialogs; **duplex agents (Moshi, GPT-4o Realtime, Gemini Live)
  have no "between turns"** — no clean utterance boundary to trigger on, no
  idle gap to warm a cache in.

Meanwhile no generative-retrieval system accepts spoken queries at all
(survey verified 2026-07: GENIUS covers image/text queries; MSEB baselines are
all ASR/embedding cascades; audio appears only on the candidate side, e.g.
GTA). The empty square this proposal claims:

|                      | turn-final        | within-turn / streaming |
|----------------------|-------------------|-------------------------|
| **cascade (ASR→text)** | S2R, MSEB baselines, VoiceAgentRAG | streaming-ASR polling (baseline we build) |
| **audio-native**       | streaming dense (baseline we build) | **SimulGR (this work)** |

Motivation is borrowed, not argued: S2R is in production, MSEB shows WER does
not predict retrieval quality (so fixing the cascade ≠ fixing retrieval), and
VoiceAgentRAG quantifies how much hiding retrieval latency matters. This is a
**model paper**: the contribution is the retriever, evaluated on a public
benchmark (SVQ/MSEB), with one systems demo at the end.

### Contributions

1. **First speech-native streaming generative retriever** (spoken query →
   semantic ID, incremental decoding during speech).
2. **Anytime Semantic IDs (A-SID)**: identifier space co-designed with
   streaming — coarse tokens decodable from short audio prefixes, fine tokens
   from complete evidence.
3. **Content-gated commitment via a learned ⟨defer⟩ token**: the model commits
   on *information*, not elapsed time — "I'd like to know…" prefixes produce
   deferral, not thrash (§4.3, §4.5).
4. **Cross-modal self-distillation recipe**: one decoder-only model whose text
   branch (transcript → SID) teaches its audio branch (audio prefix → SID) —
   importing the SpeechDPR lesson (all its margin came from text KD) into GR.
5. **Streaming-retrieval evaluation protocol**: MRR-vs-audio-fraction curves,
   time-to-first-correct-commit, commitment stability, prefetch hit/waste at
   matched latency; plus a voice-agent-loop demo measuring grounded-response
   onset latency against a VoiceAgentRAG-style turn-final system.
6. **(Gated) SpokenQG — expansion for speech retrieval, done right**: the
   first systematic answer to "what is doc2query for spoken queries?" —
   spoken-register QG as the workhorse, and text-only expansion carried
   across modalities by the shared decoder (does expansion need audio at
   all?) (§4.4). Content and acoustics are always decoupled — the lesson of
   the thesis's UnitQG dead end.

### Research questions

- **RQ1 (quality):** Does audio-native GR over semantic IDs match ASR-cascade
  and streaming-dense retrieval at full-query on SVQ?
- **RQ2 (streaming):** How much final quality is reachable from partial audio,
  and do A-SID + prefix distillation beat naive prefix training on the
  quality/stability-vs-latency frontier?
- **RQ3 (commitment):** Can defer-calibrated commitment convert partial
  predictions into prefetch actions with high hit rate and low waste, reducing
  end-to-end grounded-response onset latency?

---

## 2. Why not the thesis setup (what changes and why)

| Thesis (SLUE SpeechGR)            | SimulGR                                   | Reason |
|-----------------------------------|-------------------------------------------|--------|
| Flan-T5-base encoder–decoder      | decoder-only LLM (Qwen3 class)            | T5's bidirectional encoder must re-encode the whole query on every new audio chunk — O(T²) over a stream, no state reuse. A causal decoder-only model prefills audio incrementally into its KV cache; each retrieval probe reuses it. Streaming is *architecturally native*, not bolted on. |
| URL-encoded title-string docids (up to 39 tokens, text-orthographic) | A-SID: 4 fixed tokens, coarse→fine, built from passage-embedding residual quantization | Short = cheap repeated probing. Hierarchical = partial audio yields a valid partial ID (a prefetchable cluster). Semantic-in-embedding-space, not in spelling. |
| Pure textless (HuBERT units only) | text branch as first-class citizen        | Thesis + SpeechDPR ablations both show the gap *is* text supervision (19.73 w/ KD vs 0.04 w/o). SVQ ships transcripts; refusing them is a handicap, not a contribution. Textless becomes an *ablation row*, giving the thesis narrative closure. |
| Offline, full-query decoding      | incremental probes + commitment policy    | The point of the paper. |

---

## 3. Data

**Primary: SVQ / MSEB passage retrieval** (Google, arXiv 2602.07143). ~177k
spoken utterances, 26 locales / 17 languages, 4 acoustic conditions per query
text, transcripts included; in-language corpus ≈ 271k Wikipedia passages
(text). Metric: MRR. Scope order:

- **Phase A:** English locales (en_*), clean + noisy conditions.
- **Phase B:** multilingual with a *shared* A-SID space (IDs built from a
  multilingual embedder ⇒ cross-lingual GR for free — attacks MSEB's
  "embeddings bound by language" finding).

**Secondary: SLUE-SQA-5** — continuity with thesis, known baselines, sanity
transfer check. Not the headline. **Deferred (decision 2026-07-14): no SLUE
runs until SVQ results are proven (G1 passed).**

**Verified (2026-07-14):** SVQ lives at `hf.co/datasets/google/svq` —
CC-BY-4.0, 33.5 GB, 177,352 utterances with audio, transcript, locale,
condition, and speaker metadata. **It ships as a single undivided evaluation
set — no official train split.** Consequences:

**Training/eval protocol (decided 2026-07-14): zero-shot-first — never train
on SVQ audio.** SVQ's card frames it as an undivided evaluation set and
MSEB's baselines are zero-shot systems; we adopt the same contract and turn
the missing train split into a strength:

- **Supervised audio queries come from TTS, not SVQ.** SVQ's query texts are
  drawn from XTREME-UP's *validation/test* sets — so XTREME-UP's **train**
  questions (with gold passages) are in-distribution-family and leakage-free
  *by construction*. TTS them (multi-speaker, multi-style) with condition
  augmentation mimicking SVQ's four conditions (MUSAN noise/media/traffic +
  room IRs).
- **Verified train volumes (XTREME-UP Table 1, 2026-07-14):** in-language
  retrieval train = **29,683** question–passage pairs (6 UL + 3 HL
  languages; HL training data is *not* size-capped); in-language QA train =
  **59,559** (gold-passage format ⇒ reusable as retrieval supervision);
  cross-language retrieval/QA train = 13,270 / 22,544. English's exact share
  needs the M0 download to count, but the human-written pool is order 10⁴ —
  enough for the in-distribution slice, not for corpus coverage. The
  **corpus is confirmed**: XTREME-UP's retrieval index = **271k in-language
  passages** (447k English for cross-language), the same index MSEB uses,
  with distractors drawn from the same articles as targets.
- **doc2query + TTS for corpus coverage:** GR needs every passage reachable
  from some query (the DSI-QG lesson); generate consistency-filtered
  pseudo-questions per passage and TTS them. Together these two sources form
  the Stage-2 training set.
- **Real speech comes from Stage 1.5, not SVQ:** LibriSpeech + SLUE-SQA-5
  train queries (free real spoken *questions*, usable for ASR/LM tasks since
  Stage 1.5 teaches hearing, not SVQ's corpus ids).
- **SVQ itself: dev/test only.** One speaker- AND text-disjoint dev/test
  partition with published split lists; dev for model selection and θ
  calibration, test touched once per paper table. The TTS→real gap is
  *measured* (TTS-dev vs real-dev), not hoped away.
- Payoff: headline numbers follow MSEB's zero-shot spirit (comparable in
  kind to the paper's cascade baselines), immune to "trained on the eval
  benchmark" reviews; an optional "+in-domain adaptation" row (fine-tuning
  on a train share of an SVQ split) becomes an *analysis*, not a dependency.
- **Corpus location:** the passage KB derives from XTREME-UP retrieval/QA;
  confirm the exact corpus files via the MSEB repo at M0.

**TTS engine plan and budget (decided 2026-07-14, revisit after M0 pilot).**
Precedent: training spoken-QA systems on TTS with human-speech test sets is
standard (NMSQA train = Amazon Polly TTS with human test; Spoken-SQuAD =
Google TTS; HeySQuAD ships human + machine versions). Our recipe:

- **Bulk engine — Kokoro-82M** (Apache-2.0, 54 voices / 8 languages, ~36×
  realtime on a T4, faster with batching on the 4090): synthesizes the
  doc2query corpus-coverage set. Budget: 271k passages × 2 questions ≈ 540k
  utterances ≈ ~600 h audio ⇒ **roughly 10–20 GPU-hours** on the 4090 —
  an overnight job, ~$0.
- **Diversity engine — CosyVoice2-0.5B (Apache-2.0) or Qwen3-TTS (Jan 2026,
  10 languages, 3-s voice cloning)**: zero-shot voice cloning with reference
  prompts sampled from LibriSpeech train-other-500 (~1.1k speakers) /
  Common Voice ⇒ thousands of synthetic voices (SVQ itself has 700
  speakers). Used for the high-value XTREME-UP train questions and a slice
  of doc2query; slower (~order 2–5× realtime), fine at 10⁴–10⁵ scale.
- **Two-engine mixing** guards against single-engine acoustic artifacts;
  condition augmentation (MUSAN + room IRs, CPU, offline) multiplies
  conditions at zero TTS cost.
- **Rejected:** XTTS-v2 (non-commercial CPML license; permissive
  alternatives exist); commercial APIs for bulk (≈33M chars ⇒ roughly
  $500–1,000 at typical neural-TTS pricing — pointless when local is free;
  at most a small API-voiced quality subset later).
- **M0 pilot gate:** synthesize ~10k utterances first, train a small Stage-2
  pilot, and measure the TTS→real gap on SVQ real-dev *before* committing
  to full-scale synthesis.

**Scope note — static corpus:** GR requires (re)indexing to admit new
passages; this proposal fixes the corpus (the XTREME-UP/MSEB index is
static). Continual/incremental indexing is its own literature (MixLoRA-DSI;
parametric memory heads) and explicitly out of scope.

**Considered and rejected — AudioCaps/Clotho:** those are audio↔caption
retrieval datasets (text-caption queries, environmental-sound *items*) —
the candidate-side-audio square that GTA (Interspeech 2025) already
occupies, with tiny corpora (~5k/~50k clips), no transcripts, and no spoken
queries; the streaming/voice-agent thesis has no meaning there. Wrong task,
occupied territory.

---

## 4. Model design

### 4.1 Backbone: decoder-only

```
 audio stream ──► Mimi tokenizer (causal, 12.5 Hz) ──► audio tokens
                                                          │ append (incremental prefill, KV cache)
 ┌────────────────────────────────────────────────────────▼─────────────┐
 │  Qwen3-0.6B / 1.7B  (single decoder, shared weights, 3 input modes)  │
 │                                                                      │
 │  index:      [DOC]  passage text            ⇒  [SID] c1 c2 c3 c4     │
 │  text-query: [TXT]  transcript              ⇒  [SID] c1 c2 c3 c4     │
 │  audio-query:[AUD]  audio tokens (prefix)   ⇒  [SID] c1 c2 ⟨defer⟩   │
 └──────────────────────────────────────────────────────────────────────┘
                       │ probe = ≤4 constrained decode steps from cache
                       ▼
        trie-constrained beam over A-SID space (+ ⟨defer⟩ at each level)
```

- **Base model:** Qwen3-0.6B first, with a usability gate (G0/G1) before any
  scale-up; Qwen3-1.7B only for headline numbers after the recipe is proven
  (LoRA or rented A100 if 24GB is tight).
- **This is a custom token-LM stack, not Qwen-Audio.** Qwen-Audio /
  Qwen2-Audio / Qwen2.5-Omni do **not** use Mimi or any discrete codec: they
  feed *continuous* features from a Whisper-style encoder through an adapter.
  We take the plain *text* Qwen3 and extend its vocabulary with Mimi
  codebook-1's 2048 codes — exactly the thesis pattern (HuBERT units →
  Flan-T5 token lookup), so the tooling experience transfers directly.
- **Why not Qwen2-Audio/Omni as the base — honest tradeoff.** On raw
  full-audio quality ceiling, Qwen2-Audio-7B / Qwen2.5-Omni-3B would likely
  be *stronger*: their pretrained speech–text alignment is worth a lot at
  SVQ's data scale. We still start custom because:
  (i) **iteration** — full fine-tuning of 0.6B on the 4090 vs LoRA-only on
  3–7B (slower, costlier, rented-GPU-bound);
  (ii) **streaming purity** — Mimi is frame-causal at 80ms; Whisper-style
  encoders are block-bidirectional (Omni streams in ~2s blocks), which
  coarsens and muddies the MRR-vs-ρ curves that are the paper's core
  evidence;
  (iii) **the audio interface is a research variable** — A-SID, the
  curriculum, and ⟨defer⟩ all interact with tokenization; we need full
  control of it;
  (iv) the duplex/Moshi narrative is token-native.
  The hearing gap is mitigated by an explicit Stage-1.5 (below), and hedged
  twice: if G1 diagnoses *perception* as the bottleneck, swap the front-end
  to a pretrained encoder (option B) with the recipe unchanged; and compute
  permitting, run the full recipe on Qwen2.5-Omni-3B (LoRA) as a
  portability/ceiling appendix row.
- **Audio interface (primary): Mimi** discrete tokens — the one mainstream
  codec that is causal/streaming by construction (80ms frames, 12.5 Hz), with
  a WavLM-distilled (semantic) first codebook. Use codebook 1 only (10s query
  ≈ 125 tokens); ablate +codebook 2. Revives the stalled `feat/mimi`
  scaffold. Vocabulary grows by 2048 Mimi tokens + 3×256 SID tokens +
  specials — trivial (`resize_token_embeddings`).
- **Audio interface (option B):** small causal conv/downsample adapter on a
  streaming encoder outputting ~6–12 continuous embeddings/sec. Higher
  ceiling, more moving parts. Decide by an M1 side-by-side on 10% data.
- **Dialog-ready by construction:** a decoder-only LM conditions on prior
  conversation turns in-context with zero architecture change —
  context-dependent queries ("what about *its* population?") become a Phase B
  extension, impossible in the T5 setup.

### 4.1.1 Precedent: decoder-only GR is proven; the port pitfalls are known

Survey (2026-07-14) grounding the backbone choice:

- **Zhang et al. 2025** (arXiv 2509.22116, ICT-CAS/UvA, "Does Generative
  Retrieval Overcome the Limitations of Dense Retrieval?") build **all** GR
  models on **Qwen3-0.6B** — our exact base — with residual-quantization
  codebook ids (length 6 × 256 codes) and title ids, trie-constrained
  decoding, full fine-tuning, on NQ (300k docs) and MS MARCO (1M). Findings
  that favor this proposal: under corpus scaling GR degrades ~2–3× slower
  than dense retrieval (NQ Hit@1: −3.3 vs −6.9); GR gains ~5% from added
  parameters where DR stays flat; GR's globally-normalized objective avoids
  DR's calibration drift. Qwen3-0.6B + RQ ids + trie is exactly our Stage-1
  configuration, already validated on text.
- **STATIC** (arXiv 2602.22647): decoder-only GR (Gemini-based 3B, Gemma 1B)
  with RQ-VAE semantic ids (L=8 × 2048 on YouTube; L=4 × 256 on Amazon)
  **deployed on YouTube feeds serving billions**; trie-constrained decoding
  costs 0.033 ms/step (0.25% of inference) with their accelerator-friendly
  trie. Industrial existence proof for decoder-only + semantic-ID + trie.
- **LC-Rec** (Llama-7B + RQ-VAE id tokens added to the vocabulary, six
  alignment tasks, full fine-tuning) and **Self-Retrieval** (NeurIPS 2024,
  StableLM-3B, full FT, corpus internalized end-to-end) round out the
  precedent set.

**Full FT vs LoRA:** every successful *initial* GR training above is full
fine-tuning. Mechanism: the indexing task is corpus **memorization**, and
low-rank updates are worst-suited to exactly that — Biderman et al. 2024
("LoRA Learns Less and Forgets Less") show standard-rank LoRA substantially
underperforms full FT on new-knowledge acquisition (full-FT updates are
effectively 10–100× higher rank). LoRA appears in the GR literature only for
*continual updates* of new document slices on top of a fully-trained base
(MixLoRA-DSI, EMNLP 2025). Consequence: Qwen3-0.6B **full FT** on the 4090
is the right regime; any future LoRA run on a 3B+ backbone must use high
rank and unfreeze `embed_tokens`/`lm_head`.

**Decoder-only port checklist** — each a known *silent* failure mode vs the
T5 setup (candidate explanations for the prior Llama attempt):

1. **New id/unit tokens under PEFT**: default LoRA targets attention/MLP
   only; `embed_tokens` + `lm_head` stay frozen, so new token rows never
   learn (`modules_to_save` or full FT required).
2. **New-token initialization**: even mean-init collapses new tokens into a
   degenerate subspace that fine-tuning cannot fully recover (GTI, arXiv
   2604.02324) — init each SID token from the mean embedding of its
   cluster's passage texts (free, grounded init).
3. **Loss-mask the prompt** (`labels = -100`): otherwise loss is dominated by
   reproducing the query/passage, not predicting the docid (T5 gets this for
   free architecturally).
4. **Trie offset**: `prefix_allowed_tokens_fn` receives prompt+generated for
   decoder-only; T5-era trie code indexing from position 0 silently produces
   garbage constraints — slice at prompt length.
5. **Left-padding** for batched generation (right-padding silently corrupts
   decoder-only outputs).
6. **Keep the indexing task + pseudo-queries**: query→id pairs alone don't
   cover the corpus (DSI recipe requirement, backbone-independent).
7. **BPE quirks**: Llama/Qwen digit- and URL-tokenization inflate string ids
   — moot here since A-SID tokens are dedicated vocabulary entries.

### 4.2 Identifier: Anytime Semantic IDs (A-SID)

**Length and capacity (why 3 levels × 256 + 1 dedup = 4 tokens).** Capacity
is combinatorial, not additive: 256³ = **16.7M leaf cells** for 271k passages
(≈62× headroom), with the 4th token disambiguating any residual collisions
inside a leaf. Deeper is not free:

- each extra level is one more *sequential* decode step at every streaming
  probe (probes fire every ~0.3–0.6s for the whole session);
- ID decoding is autoregressive — every extra level is another chance to fall
  off the gold path and needs beam width to cover it (the thesis measured
  this length tradeoff directly: atomic 1-token ids won Hit@1, long string
  ids won Hit@20);
- deeper-but-narrower (e.g. 6×32 = 1.07B) makes each level carry less
  information and level 1 uselessly coarse for prefetch (32 cells ≈ 8.5k
  passages each).

At 3×256 the hierarchy is practically useful: level-1 commit ≈ 1/256 of the
corpus (~1k passages — shard-warming granularity), level-2 ≈ 4 passages
(direct prefetch granularity), level-3 near-singleton. **Pre-check at M0,
offline, before any training:** quantize the corpus, measure leaf collision
counts and level balance (use balanced k-means if skewed); if collisions are
bad, fall back to 4×256 (4.3B cells) at the cost of one decode step. The
binding constraint is balance and predictability, not capacity.

**Construction** (offline, once; **we train this quantizer ourselves** — it
is corpus-specific by construction, so nothing pretrained exists to reuse;
TIGER/RIPOR/LMIndexer all train their own. It quantizes 271k *embedding
vectors*, not audio — residual k-means (faiss) is minutes of compute; the
RQ-VAE variant needed for the prefix-predictive loss is a ~1–5M-param model,
under an hour on the 4090. Mimi, by contrast, is never trained — frozen
Kyutai checkpoint):

1. Embed all passages with a frozen text embedder (multilingual, e.g.
   Qwen3-Embedding-0.6B or GTE; pick at M0 by cascade-baseline quality).
2. Residual quantization (RQ-VAE or residual k-means) per the geometry above.
3. **Query-aware cell assignment:** quantize a blend
   `z' = α·z_passage + (1−α)·mean(z_queries/pseudo-queries)` so coarse cells
   cluster by *what queries sound like asking*, and absorb acoustic/wording
   variation at level 1 while fine levels carry the discriminative load.

**How "anytime" is actually achieved (RQ alone does NOT give it).** Vanilla
RQ-VAE is coarse-to-fine in *embedding-reconstruction* space; nothing aligns
level-1 codes with *what is audible early in a query*. RQ is the
initialization, not the mechanism. Three enforcement layers:

1. **Model-side (primary): level-fraction curriculum + prefix distillation**
   (§4.3). For prefix fraction ρ, supervise level k with weight w_k(ρ) —
   level 1 trained hard from short prefixes, levels 3–4 only near ρ=1. This
   trains p(c1 | audio_≤ρ) to marginalize over the still-uncertain leaf; it
   works iff coarse cells are predictable-in-principle from partial evidence.
2. **Codebook-side (cheap): the query-aware blend** in construction step 3
   moves cell boundaries toward query space.
3. **Codebook-side (novel, the real identifier contribution):
   prefix-predictive residual quantization** — train the quantizer with an
   auxiliary objective that ρ-truncated *transcript* embeddings must predict
   the coarse codes:
   `L = L_recon + λ · Σ_ρ CE(c_{1..k(ρ)} | E_text(transcript_≤ρ))`.
   The ID space is thereby *optimized* so early evidence identifies coarse
   cells. Offline job over 271k passages + truncated-query embeddings —
   minutes-to-hours, no LLM in the loop.

**Verification before committing (M0):** define **prefix predictability** =
accuracy of a linear probe predicting the level-k code from ρ-truncated query
embeddings. Compare plain RQ vs blended vs prefix-predictive RQ offline. This
both selects the codebook and yields the paper's analysis figure. Existing
semantic-ID work (TIGER, RIPOR, LMIndexer) is static w.r.t. the query —
time-aligned, prefix-predictive ID construction is the new axis.

### 4.3 Training recipe

**Stage 0 — codebook** (offline): §4.2.

**Stage 1 — text GR:** multitask CE on `[DOC] passage ⇒ SID` and
`[TXT] transcript ⇒ SID` (+ optional filtered doc2query pseudo-queries — with
a consistency filter this time; unfiltered QG is a known dead end from SLUE).

Why text first (three reasons, in order of force):

1. **Diagnostic isolation (principles 3-a/3-b):** text→SID proves the data,
   the ID space, the backbone, and the trie-decoding pipeline with *zero*
   audio variables, against a known reference (the embedder that built the
   IDs). If audio-first failed we could not attribute the failure. This is
   exactly gate G0 — it proves the setup works before audio spend.
2. **Forced by the recipe:** Stage 2's distillation teacher *is* the text
   branch — it must exist and be good before Stage 2 can run at all.
3. **Transfer:** the thesis's single biggest lever was pretraining the
   backbone before GR (14.19 → 18.45 Hit@20); same shape here — audio
   training becomes *alignment to an existing retrieval competence* rather
   than learning retrieval and hearing simultaneously, which is also where
   SpeechDPR's entire margin came from (text-teacher KD).

**Stage 1.5 — hearing pretraining (custom-stack mitigation):** the text-init
backbone has never heard audio, and SVQ alone is small for learning speech
perception from scratch. Before retrieval alignment, multitask on public
speech tokenized by frozen Mimi: (a) ASR-style `audio tokens ⇒ transcript`
(LibriSpeech 960h + SLUE-SQA-5 train queries — real spoken questions; never
SVQ audio, per the §3 protocol), (b) next-token LM on Mimi streams.
This is the direct heir of the thesis's single biggest lever (unit-LM
pretraining: 14.19 → 18.45 Hit@20), now with far more data available since
Mimi tokenization of LibriSpeech is a cheap one-off pass.

**Stage 2 — audio alignment + streaming:**

- CE on `[AUD] full audio ⇒ SID`.
- **Prefix-consistency distillation:** sample ρ ∈ {0.25, 0.5, 0.75, 1.0} (by
  forced-alignment word boundaries, one WhisperX/CTC pass over train queries);
  student sees `[AUD] audio_≤ρ`, teacher = same model's text branch on the
  *full* transcript (frozen or EMA);
  loss = Σ_k w_k(ρ)·CE + β·KL(p_audio(SID | prefix) ‖ p_text(SID | full)).
- **⟨defer⟩ supervision (content-gated commitment):** run the text teacher on
  the *truncated* transcript at ρ. If its top-1 level-k token already equals
  the final one → supervise token; else → supervise ⟨defer⟩ at level k and
  stop. Fully automatic hindsight labeling, no annotation. A prefix that is
  pure carrier phrase ("I'd like to know…") gets ⟨defer⟩ at level 1 — the
  model learns that filler carries no retrieval information, which directly
  answers the "'i want to know' prefix has no meaning" problem: gate on
  information, not on elapsed time (semantic endpointing, not wait-k).
- **Carrier-phrase augmentation:** SVQ is read speech; real queries arrive
  wrapped in filler. Prepend sampled spoken carriers (TTS or concatenative
  bank: "hey, can you tell me", "I was wondering", zh equivalents in Phase B)
  with ⟨defer⟩ targets over the carrier region; also build a filler-augmented
  test split for robustness reporting.
- **Cross-condition consistency:** symmetric KL between SID posteriors at
  matched ρ for the same query under different acoustic renderings. Under
  the zero-shot protocol the training pairs come from our own condition
  augmentation of TTS queries; SVQ's native same-text-under-4-conditions
  pairs are used on the *evaluation* side as a condition-invariance metric.

**Default: CE-only, no negatives.** Indexing CE, retrieval CE,
prefix-distillation KL, and consistency KL are all target-based; GR's CE is
globally normalized over the id space — every other docid is an implicit
negative at each step (Zhang et al. 2509.22116). This is a **default based
on our prior evidence, not a principle**: the thesis's bolt-on margin
ranking loss cost −1.35 Hit@20 *on SLUE/T5* — one setup, possibly
confounded — while LTRGR and RIPOR report gains from ranking-aware
objectives in text GR. Revisit as a *designed experiment* if the symptom
they target appears: full-audio recall fine (Hit@10/20) but ranking weak
(Hit@1/MRR). Not a default ingredient.

**Stage 3 — commitment calibration:** no separate policy network needed —
commit level k when p(⟨defer⟩ at k) < θ; sweep θ for the latency/waste
frontier. Optional learned head trained on hindsight stability labels as an
ablation. Committed prefixes constrain the trie (monotone), with one allowed
repair if the posterior diverges hard (report repair frequency).

### 4.4 Speech-native document expansion (SpokenQG — gated contribution)

**The gap:** pseudo-query generation is *the* load-bearing ingredient of GR at
scale (DSI-QG; the "how does GR scale" lesson), yet in speech retrieval it
exists only as a cascade: text doc2query → TTS. No speech-native document
expansion exists (verified 2026-07: doc2query / Doc2Query-- / Doc2Query++ /
query2doc are all text-side). The thesis's UnitQG was the closest attempt
anywhere and failed **unfiltered** on SLUE (16.66 < 18.45 Hit@20); it lacked
three ingredients this setup supplies: a strong pretrained backbone (Stage
1.5), text-side grounding, and mandatory filtering.

**Why UnitQG failed, structurally:** it coupled two problems of very
different difficulty — *query content* (what to ask; a text-competence
problem) and *acoustic realization* (how it sounds) — into one generation
task in a lossy unit space, with no filter between generator noise and
retriever training. The alternatives below all **decouple content from
acoustics**; none re-attempts speech-authoring-from-scratch.

- **A — Spoken-register QG → TTS (primary, M1 data pipeline):** content from
  a text LLM (doc2query, consistency-filtered), *spokenness injected at the
  text level* — fillers, hesitations, restarts, entity-early vs entity-late
  phrasings — then TTS + condition augmentation. Free synergies: filler
  spans known at generation time ⇒ **exact ⟨defer⟩ supervision without
  forced alignment** (replaces carrier-phrase concatenation); entity-position
  control = designed stress test for anytime commitment.
- **B — Text-only expansion via the shared decoder (compute-saver and
  analysis headline):** don't synthesize audio for most pseudo-queries at
  all. Feed generated questions through the **text branch** (`[TXT] question
  ⇒ SID`) and let Stage-2 cross-modal distillation transfer the query-form
  competence to the audio branch. Speech enters training only where it must:
  hearing (Stage 1.5) and alignment (a modest TTS/real set). Testable as an
  **audio-pseudo-data fraction sweep** (0% → 100% of pseudo-queries
  TTS'd): if the curve is flat, "audio pseudo-queries are unnecessary" is
  itself a headline finding — and expansion becomes nearly free.
- **C — Token-space generation (exploratory only, demoted):** the UnitQG
  descendant, kept only as a late, cheap ablation and only in its
  content-anchored form first — *re-synthesis* of an existing query's Mimi
  tokens (speaker/prosody perturbation, codec re-voicing), never authoring
  content in token space. Full self-expansion (`passage ⇒ query tokens`) is
  attempted, if at all, after A/B conclusions, behind a round-trip filter,
  and a negative result ships as an ablation row.

**Cut from this section — real-speech pseudo-labeling is not expansion:**
retrieving corpus passages for existing spoken questions (e.g. SLUE train)
adds query-side pairs but **zero passage coverage** — it never touches the
expansion bottleneck. It is an acoustic-domain-adaptation tool, and Stage
1.5 (real-speech hearing) + text-teacher distillation already carry that
load; it survives only as a registered conditional fallback in §6.

**Gate and ordering:** A is the primary method — load-bearing and
precedented. B is a near-free training-mixture ablation asking the novel
question (does expansion need audio at all?). C is exploratory. The pipeline
never depends on B or C working (principles 3-c).

### 4.5 Inference loop

Per ~320–640ms audio chunk: append Mimi tokens to KV cache → probe = ≤4-step
trie-constrained beam (beam ~10) from the cached state (few ms at 0.6B on a
4090) → allowed set at each level = trie continuations ∪ {⟨defer⟩} → emit
committed prefix ⇒ downstream prefetch (fetch/prerank the cluster's passages,
warm the LLM context). Reuses the thesis `prefix_allowed_tokens_fn`
machinery nearly unchanged.

---

## 5. Evaluation protocol

**Metrics.** Full-audio MRR (MSEB-standard); MRR@ρ curves (25/50/75/100%);
time-to-first-correct-commit (earliest audio time the committed coarse/full ID
is correct and never flips); flip rate; prefetch hit@k vs wasted-prefetch
fraction at matched mean commit latency; compute per update (RTF).

**Baselines** (matched passage set, matched audio budget, ~0.6B class where
possible):

1. **Turn-final cascade:** Whisper-large-v3(-turbo) → frozen text embedder →
   FAISS flat over 271k. Reference ceiling; report paper's Gemini-embedding
   numbers as context only.
2. **Streaming cascade (polling):** streaming/partial Whisper hypotheses per
   chunk → same embedder → re-search. The "just poll dense" rival.
3. **Streaming dense audio (S2R-style):** audio query tower (init from a
   speech encoder) contrastively trained against *frozen* passage embeddings,
   with the same prefix sampling. The strongest fair rival — beat this and
   RQ1/RQ2 stand.
4. **VoiceAgentRAG-style semantic cache** on top of (1) — systems baseline for
   the demo only (its own eval used a 12-doc synthetic KB; not a quality bar).

**Ablations:** enc-dec (T5) vs decoder-only on identical data; title-string /
atomic / A-SID; −⟨defer⟩ (fixed wait-k cadence instead); −prefix distillation
(prefix CE only); −text branch (**the textless row** — quantifies exactly what
dropping the constraint bought, closing the thesis arc); −cross-condition
consistency; Mimi cb1 vs cb1+2; **pseudo-query source** (A TTS-cascade vs
B text-only+distillation vs C token-space, plus the audio-pseudo-data
fraction sweep); (compute permitting) full recipe on Qwen2.5-Omni-3B + LoRA
— portability and quality-ceiling check.

**End-to-end demo (RQ3):** a minimal voice-agent loop (VoiceAgentRAG-style
event loop or simplified duplex mock): measure grounded-response onset latency
(user speech end → first grounded token) with SimulGR prefetch vs turn-final
cascade — the "the chatbot doesn't wait" number.

---

## 6. Kill-gates and fallbacks (principles.md 3-c)

| Gate | Test | If failed |
|------|------|-----------|
| **G0** (M0) | Stage-1 *text*→SID MRR ≥ ~85% of the dense embedder that built the IDs, on SVQ-en transcripts | ID space or GR scaling is broken — fix identifiers before any audio work. Do not proceed. |
| **G1** (M1) | Full-audio SimulGR ≥ streaming-dense baseline (3), and within striking distance of cascade (1) | Diagnose which gap: if text→SID is strong but audio lags ⇒ **perception bottleneck** → swap front-end to a pretrained encoder (option B / Omni encoder, LoRA), recipe unchanged. If GR itself lags dense ⇒ pivot to **hybrid**: GR coarse levels as streaming prefetch router + dense rerank inside the cluster. Still a coherent paper either way. |
| **G2** (M2) | A-SID + distillation dominate naive prefix-CE on MRR@ρ and flip-rate curves | Mechanism contribution fails → diagnose (teacher quality? alignment?) before adding Stage 3. |
| **Scoop watch** | arXiv alert: "speech generative retrieval", "spoken query retrieval", MSEB citations | Escalate the (currently deferred) preprint decision immediately. |

Prior negative results — don't re-enter *blindly* (empirical observations
from the SLUE/T5 setup, not laws; each needs a designed reason to retry):
unfiltered unit-QG augmentation, bolt-on margin ranking loss (contrast:
LTRGR/RIPOR gains in text GR — retry only if Hit@1/MRR lags recall), window
Q-Former, VAD trimming.

Registered conditional fallback: if a large TTS→real retrieval gap persists
on SVQ real-dev *despite* Stage 1.5 + distillation, build a small
real-audio→SID adaptation set by pseudo-labeling real spoken questions
(SLUE-SQA-5 train ≈ 46k, HeySQuAD) against our corpus — positive-pair
labeling only; this is domain adaptation, not expansion (§4.4).

---

## 7. Feasibility (compute/disk)

- Qwen3-0.6B full FT (bf16, flash-attn, seq ≤ 640): fits 24GB 4090 (Joyboy).
  1.7B: LoRA on 4090 or rent A100 for headline runs.
- Mimi tokenization of 177k short utterances: single GPU-day order; store
  packed like `slue-sqa-code-*` (~GBs).
- 271k-passage index task ≈ same order as thesis corpus ×6 with 4-token
  labels (shorter than the 39-token title strings) — cheaper per example.
- Probe decoding: ≤4 steps × beam 10 at 0.6B ⇒ real-time capable on-device
  class; report RTF.
- Disk: SVQ audio + caches on `/storage/ricky` (verify footprint at M0).

---

## 8. Milestones (16 weeks)

| Wk | Milestone | Output / gate |
|----|-----------|---------------|
| 1–2 | **M0** data + IDs + text GR | SVQ download/EDA, dev/test split lists, corpus + XTREME-UP train verification, TTS-pipeline pilot, embedder pick, A-SID codebook + probes, Stage-1 model. **G0.** |
| 3–6 | **M1** audio branch | Mimi pipeline (revive `feat/mimi`), expansion data gen (A), Stage-2 CE, full-audio MRR vs baselines 1 & 3. **G1.** |
| 7–10 | **M2** streaming core | forced alignment, prefix distillation, curriculum, ⟨defer⟩; curves + stability. **G2.** → paper core; arXiv-able. |
| 11–13 | **M3** commitment + demo | θ sweeps, prefetch hit/waste, filler-augmented test, agent-loop onset-latency demo. |
| 14–16 | **M4** extension + writing | expansion-source analysis (B fraction sweep; C exploratory), multilingual shared-ID phase B (stretch), ablation table, paper draft. |

Venues (check exact deadlines): ICASSP 2027 (~Sept 2026 — M1-scope short
paper possible), Interspeech 2027 (~Mar 2027 — full scope), ACL ARR any
cycle. Preprint timing: decision deferred (2026-07-14); revisit at M2, or
immediately if the scoop-watch fires.

---

## 9. Related-work deltas (one line each)

- **Thesis SpeechGR:** textless T5 + title-string ids, offline, SLUE →
  SimulGR changes backbone, ids, supervision, and adds the streaming axis.
- **GENIUS (CVPR'25):** universal *image/text*-query GR, non-streaming — no
  audio, no time dimension.
- **Google S2R:** speech→retrieval, *dense*, turn-final, closed — validates
  the task, doesn't touch GR or streaming.
- **VoiceAgentRAG:** turn-based cascade + between-turn cache — motivation and
  systems baseline; structurally inapplicable to duplex.
- **TIGER / RIPOR / LMIndexer:** semantic IDs, static queries — A-SID adds
  time-aligned coarse-to-fine supervision + defer.
- **SimulST (wait-k, adaptive policies):** simultaneous decoding for
  *translation* — we import the framing; retrieval's target (a 4-token ID)
  makes anytime commitment far more tractable than full text generation.

---

## 10. Decisions (locked 2026-07-14)

1. **Backbone:** Qwen3-0.6B, gated — prove usable (G0, then G1) before any
   scale-up to 1.7B.
2. **Audio interface:** Mimi (codebook 1 primary; +cb2 as ablation).
3. **Scope:** English locales first; multilingual is Phase B.
4. **SLUE-SQA-5:** kept as secondary benchmark, but frozen until SVQ results
   are proven — no SLUE runs before G1 passes.
5. **Preprint:** decision deferred; revisit at M2 or on scoop-watch trigger.
