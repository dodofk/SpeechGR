# Plan

This execution plan turns the agreed SpeechGR direction into a Mimi-first implementation roadmap with explicit stages, detailed DocID design, transcript-free pseudo-query mining, and a later decoder-only branch. The goal is to make the next implementation pass concrete enough that another coding agent can pick up individual stages without re-litigating the architecture.

## Goals

- Build a speech-native generative retrieval pipeline without depending on the visible legacy `qg.py` path.
- Make Mimi the first-class input representation.
- Implement a hierarchical DocID that is easier to generate than a flat arbitrary ID.
- Use a fully transcript-free pseudo-query mechanism.
- Keep the first baseline simpler than `dense retriever -> build DocIDs -> train GR`.
- Keep decoder-only GR as an explicit follow-up branch, not an afterthought.

## Locked Decisions

- `Mimi first`, `WavTokenizer later if Mimi fails or underperforms`.
- `Hierarchical DocID` instead of flat DocID.
- `Transcript-free pseudo-query miner` instead of old code-to-code QG.
- `Encoder-decoder baseline first`, `decoder-only branch second`.

## Non-Goals For The First Milestone

- Do not train a new T5 from scratch on discrete audio units.
- Do not reuse the current RQ-VAE document autoencoding path as the main DocID builder.
- Do not do joint end-to-end training of tokenizer, DocID builder, pseudo-query miner, and GR model.
- Do not make WavTokenizer or decoder-only a phase-0 blocker.

## Current Repo Anchors

- Main GR path: [speechgr/cli/retrieval.py](/Users/dodofk/Research/SpeechGR/speechgr/cli/retrieval.py)
- Ranking / negative mining: [speechgr/trainer.py](/Users/dodofk/Research/SpeechGR/speechgr/trainer.py)
- Compression and T5 wrappers: [speechgr/model.py](/Users/dodofk/Research/SpeechGR/speechgr/model.py)
- Existing tokenizer wrapper to mirror: [speechgr/encoders/wavtokenizer/encoder.py](/Users/dodofk/Research/SpeechGR/speechgr/encoders/wavtokenizer/encoder.py)
- Legacy QG path to avoid depending on: [speechgr/cli/qg.py](/Users/dodofk/Research/SpeechGR/speechgr/cli/qg.py)
- Span-corruption ideas that may still be reused later: [speechgr/cli/t5_pretrain.py](/Users/dodofk/Research/SpeechGR/speechgr/cli/t5_pretrain.py)

## High-Level Architecture

```text
audio passage (30s)
  -> Mimi tokenizer
  -> passage representation / pooling
  -> hierarchical DocID builder
  -> docid_map.json

audio query or pseudo-query span (3-8s)
  -> Mimi tokenizer
  -> GR model
  -> cluster token(s) -> leaf token(s)
  -> constrained decode over valid hierarchy
```

## Detailed Doc Identifier Design

### Why Hierarchical DocIDs

Flat DocIDs force the generator to memorize arbitrary mappings. In speech this is especially brittle because the input is noisy, variable in duration, and often only partially matches the passage. A hierarchical DocID gives the model a coarse-to-fine decoding path:

- first predict the right semantic neighborhood
- then predict the exact passage identity inside that neighborhood

This is the intended meaning of `passage cluster -> passage leaf`.

### Passage Unit

Use a 30-second passage as the default indexing unit.

Recommended defaults:

- corpus passage length: 30 seconds
- corpus stride: 15 seconds for long documents, if the source corpus requires chunking
- query span length for pseudo-queries: 3 to 8 seconds

### DocID Structure

Recommended v1 format:

- `cluster stage`: 1 token
- `leaf stage`: 2 tokens

Concrete example:

```text
<cl_042> <lf1_117> <lf2_008>
```

Meaning:

- `<cl_042>`: coarse semantic region
- `<lf1_117>`: first local refinement inside the cluster
- `<lf2_008>`: exact passage-level refinement or collision resolver

Why `1 + 2` first:

- shorter than long RQ sequences
- still coarse-to-fine
- easy to decode with constraints
- enough capacity for initial corpora while keeping the target sequence short

If the corpus grows or collisions rise, move to:

- `1 cluster + 3 leaf`

Do not start with a longer sequence than needed.

### How DocIDs Will Be Implemented

#### v1 Embedding Source

Use a fixed, pooled speech embedding for each 30-second passage.

Recommended implementation choice:

1. tokenize passage with Mimi
2. convert tokens to a compact sequence representation
3. pass through a fixed or lightly adapted passage encoder
4. pool to one passage vector

For passage encoding, use the simplest stable option first:

- preferred v1: frozen speech encoder plus pooling
- only later: query-aware dense retriever embeddings

The main principle is:

- `DocID builder should consume stable semantic passage embeddings`
- `DocID builder should not depend on a reconstruction objective`

#### v1 Hierarchy Construction

Use a `coarse cluster + residual leaf` hybrid.

Step-by-step:

1. Build one pooled embedding per passage.
2. Run global coarse clustering over all passage embeddings.
3. Assign each passage a single cluster token.
4. For passages inside each cluster, fit a local residual or local clustering scheme.
5. Assign leaf tokens from that local scheme.

Recommended first hyperparameters:

- number of coarse clusters: 128 or 256
- leaf-1 size per cluster: 128 or 256
- leaf-2 size per local bucket: 64 or 128

Recommended starting point:

- `cluster = 256`
- `leaf1 = 128`
- `leaf2 = 64`

That is more than enough for the current expected scale while keeping token vocabularies modest.

#### Why Not Pure Flat Residual Quantization For v1

Pure multi-level RQ over all passages is elegant, but for the first implementation the explicit `coarse cluster + local leaf` split is easier to inspect, debug, and decode. It also makes constrained decoding simpler because the leaf vocabulary can be conditioned on the chosen cluster.

#### Collision Handling

Collision policy must be explicit from day one.

Use this order:

1. inspect collision rate after initial clustering
2. if low, keep current depth
3. if moderate, increase `leaf2`
4. if still present, append deterministic suffix token `<sx_00N>` only for collided cases

Do not treat collisions as acceptable by default.

### Constrained Decoding

Build a trie that encodes valid paths:

```text
root
  -> <cl_000>
       -> <lf1_000>
            -> <lf2_000>
            -> <lf2_001>
       -> <lf1_001>
  -> <cl_001>
       -> ...
```

Generation policy:

- position 1: only cluster tokens allowed
- position 2: only valid `leaf1` tokens under the chosen cluster
- position 3: only valid `leaf2` tokens under chosen cluster and leaf1

This should reuse and extend the existing prefix restriction approach already adjacent to the retrieval stack.

## Transcript-Free Pseudo-Query Miner

### Principle

Do not generate pseudo-queries as new token sequences. Instead, mine short audio spans that can plausibly act as queries for their parent 30-second passage.

### v1 Miner: Bootstrap Span Selection

For each indexed passage:

1. slide a short window over the 30-second passage
2. remove silent windows with VAD or simple energy filtering
3. sample 3-8 second candidate spans
4. keep N spans per passage as pseudo-queries

Recommended starting values:

- span lengths: 3s, 5s, 8s
- spans per passage: 3 to 5
- overlap allowed: yes

### v2 Miner: Boundary-Aware Selection

Improve over random crops by preferring spans around semantic or acoustic boundaries.

Possible signals:

- Mimi token distribution change
- hidden-state change in a frozen speech encoder
- pause plus high-variance continuation
- novelty score relative to neighboring windows

Use boundary-aware selection to produce more query-like spans that are coherent but still shorter than the parent passage.

### v3 Miner: Contrastive Filtering

Score candidate spans by how specifically they identify the source passage.

For each candidate span:

- compute similarity to its parent passage
- compute similarity to nearby or random other passages
- keep spans with high `parent - impostor` margin

This gives a transcript-free analogue of selecting informative queries rather than generic speech fragments.

### How Pseudo-Queries Feed Training

Use a staged curriculum:

1. real queries only baseline
2. add bootstrap pseudo-queries with reduced weight
3. add boundary-aware pseudo-queries
4. add contrastive-filtered pseudo-queries and nearby-cluster negatives

Recommended weighting:

- real query examples weight: 1.0
- pseudo-query examples weight: 0.3 to 0.5 initially

## Mimi-First Implementation Policy

### Why Mimi First

- lower token rate
- better fit for 30-second passages
- less pressure on T5 context length
- likely more semantically useful for retrieval than a higher-rate acoustic codec

### What To Do With WavTokenizer

Do not implement WavTokenizer in the main execution path now.

WavTokenizer becomes a later branch only if:

- Mimi integration fails technically
- Mimi performs materially worse than expected
- or a later ablation is needed for publication completeness

## Backbone Decision: Why Not Decoder-Only Immediately

### Why Not Immediately

The repo today is strongly seq2seq-oriented:

- retrieval path is built around encoder-decoder assumptions
- ranking logic is already adjacent to that setup
- compression wrappers in `model.py` are T5-shaped
- constrained decoding support is already closer to the encoder-decoder flow

Switching immediately to decoder-only would require:

- new causal LM wrapper
- speech-embedding projection into LLaMA/Qwen token space
- causal loss masking for DocID-only positions
- causal constrained decoding support
- new evaluation / generation plumbing

That is a lot of refactor before the real hypotheses are proven.

### Why Keep Decoder-Only On The Roadmap

The literature does not support ignoring it.

Papers and takeaways:

- `RIPOR`: strongest encoder-decoder reference for relevance-based DocIDs and coarse-to-fine retrieval.
- `NAIL`: index-learning retrieval can work with encoder-decoder and decoder-only model families.
- `Exploring Training and Inference Scaling Laws in Generative Retrieval`: LLaMA-style models outperform T5 across its studied scaling settings, so decoder-only matters for scaling.
- `Generative Retrieval as Multi-Vector Dense Retrieval`: caution that GR must justify itself beyond novelty, so the plan should validate DocID and pseudo-query benefits before a costly backbone change.

### Final Backbone Policy

- Phase 1 to 4: encoder-decoder baseline
- Phase 5+: decoder-only branch if and only if the earlier stages are stable

## Detailed Stages And To-Dos

## Stage 0: Repo Preparation And Plan Lock

### Objective

Lock the Mimi-first architecture, add missing config skeletons, and make passage indexing assumptions explicit.

### To-Do

[ ] Create repo-local plan file in `docs/`.
[x] Add a new `mimi` encoder namespace mirroring the WavTokenizer encoder layout.
[x] Define canonical passage length, stride, and split handling for corpus passages.
[ ] Define output artifact names and directories for DocIDs and pseudo-queries.
[ ] Add a stage manifest document describing what outputs each stage produces.

### Deliverables

- `docs/speechgr-execution-plan.md`
- `speechgr/encoders/mimi/`
- `configs/data/slue_sqa5_mimi.yaml`

## Stage 1: Mimi Integration

### Objective

Make Mimi the first working tokenizer path for SpeechGR.

### To-Do

[x] Implement `speechgr/encoders/mimi/encoder.py`.
[x] Add `speechgr/encoders/mimi/__init__.py` and register the encoder.
[x] Add Hydra config for Mimi precompute.
[ ] Run precompute for all required splits and corpus passages.
[ ] Record token length statistics for real 30-second passages.
[ ] Confirm the data loaders can consume Mimi-based caches end to end.

### Deliverables

- Mimi precompute cache files
- tokenizer statistics report
- config for Mimi-based retrieval experiments

### Success Criteria

- Mimi cache generation succeeds on train/validation/test/corpus
- 30-second passage lengths are measured and documented
- retrieval data pipeline loads Mimi caches without custom hacks

## Stage 2: Hierarchical DocID Builder

### Objective

Build the first stable `cluster + leaf` DocID mapping.

### To-Do

[ ] Implement `speechgr/docid/` module for offline DocID construction.
[ ] Add passage embedding extraction pipeline.
[ ] Add coarse clustering step.
[ ] Add local leaf assignment step.
[ ] Serialize `docid_map.json` and `cluster_members.json`.
[ ] Build trie-ready valid decoding paths.
[ ] Compute collision report and cluster purity report.

### Deliverables

- `speechgr/docid/builder.py`
- `scripts/build_docids.py`
- `outputs/docids/<experiment>/docid_map.json`
- `outputs/docids/<experiment>/cluster_members.json`
- `outputs/docids/<experiment>/docid_diagnostics.json`

### Success Criteria

- collision rate within target budget
- cluster prefixes restrict the candidate set meaningfully
- DocID map is loadable by the retrieval stack

## Stage 3: Minimal GR Baseline On Real Queries

### Objective

Verify that the model can generate valid hierarchical DocIDs using only real queries and Mimi input.

### To-Do

[ ] Extend retrieval config to accept hierarchical DocID labels.
[ ] Extend collator / label handling if needed for structured DocID tokens.
[ ] Add constrained decoding support for the DocID trie.
[ ] Train T5 baseline on real query -> hierarchical DocID.
[ ] Compare against flat/random DocID baselines.
[ ] Compare against a dense retrieval baseline.

### Deliverables

- baseline checkpoint
- evaluation report for flat vs hierarchical DocIDs

### Success Criteria

- high valid-DocID generation rate
- hierarchical DocID outperforms flat/random DocID on retrieval metrics

## Stage 4: Transcript-Free Pseudo-Query Miner v1

### Objective

Add bootstrap pseudo-queries without transcripts.

### To-Do

[ ] Implement random-but-non-silent span miner.
[ ] Add storage format for mined spans.
[ ] Create dataset that mixes real queries with pseudo-queries.
[ ] Retrain GR with weighted pseudo-query examples.
[ ] Compare `no pseudo-query` vs `bootstrap pseudo-query`.

### Deliverables

- `speechgr/qg/bootstrap_spans.py`
- `scripts/mine_pseudo_queries.py`
- pseudo-query manifests

### Success Criteria

- pseudo-queries improve retrieval or at least valid DocID generation
- spot checks show spans are meaningful and non-silent

## Stage 5: Transcript-Free Pseudo-Query Miner v2/v3

### Objective

Make pseudo-queries more query-like and less trivial.

### To-Do

[ ] Add boundary-aware span selection.
[ ] Add contrastive filtering.
[ ] Add curriculum schedule for pseudo-query difficulty.
[ ] Add nearby-cluster negatives.
[ ] Re-run comparisons against Stage 4 miner.

### Deliverables

- improved pseudo-query miner
- ablation report across miner variants

### Success Criteria

- v2/v3 miner improves over bootstrap spans
- pseudo-query gains are robust across multiple seeds

## Stage 6: Ranking And Hard Negatives

### Objective

Activate the repo's ranking machinery only after baseline generation is stable.

### To-Do

[ ] Reuse `DSIRankingTrainer` path for hierarchical DocIDs.
[ ] Add hard negatives based on nearby clusters or dense neighbors.
[ ] Schedule in-batch and hard-negative losses gradually.
[ ] Measure whether ranking helps Hits@1 more than it hurts stability.

### Deliverables

- ranking-enabled experiment configs
- ranking ablation report

### Success Criteria

- improved ranking metrics without collapse in valid generation

## Stage 7: Decoder-Only GR Branch

### Objective

Add a serious decoder-only comparison only after the earlier phases are successful.

### To-Do

[ ] Implement `speechgr/models/decoder_only_gr.py`.
[ ] Add speech embedding projection into a causal LM input space.
[ ] Add causal DocID loss masking.
[ ] Add constrained decoding for causal generation.
[ ] Compare on the exact same Mimi input, DocID map, and pseudo-query data.

### Deliverables

- decoder-only experiment configs
- baseline decoder-only checkpoint
- direct T5 vs decoder-only comparison report

### Success Criteria

- decoder-only matches or beats encoder-decoder enough to justify future migration

## Stage 8: WavTokenizer Fallback / Later Ablation

### Objective

Only if Mimi is inadequate, add WavTokenizer as a second tokenizer branch.

### To-Do

[ ] Define compression policy before GR.
[ ] Rebuild passage embeddings and DocIDs under WavTokenizer.
[ ] Re-run the same DocID and GR comparisons.

### Deliverables

- compressed WavTokenizer configs
- tokenizer comparison report

### Success Criteria

- WavTokenizer either beats Mimi clearly or remains an ablation-only branch

## Metrics To Track Across All Stages

- valid DocID generation rate
- Hits@1
- Hits@10
- Hits@20
- MRR
- cluster purity
- leaf collision rate
- token length distributions
- training time and inference time

## Immediate First Actions

If another agent starts implementing tomorrow, the first five concrete actions should be:

1. implement `speechgr/encoders/mimi/encoder.py`
2. add `configs/data/slue_sqa5_mimi.yaml`
3. build `scripts/build_docids.py` for `cluster + leaf` DocIDs
4. wire hierarchical DocIDs into the current retrieval path
5. build `scripts/mine_pseudo_queries.py` with bootstrap span selection only

## Do First / Postpone

### Do First

- Mimi integration
- hierarchical DocID builder
- real-query T5 baseline
- bootstrap transcript-free pseudo-query miner

### Postpone

- decoder-only backbone
- WavTokenizer branch
- complex query synthesis
- query-aware dense DocID construction
- joint end-to-end training

## Risks

- hierarchical DocID may be too coarse or too fine on the first try
- Mimi may still need some compression depending on the exact context setup
- pseudo-query spans may capture too much low-level acoustics before boundary and contrastive filtering are added
- decoder-only may become a time sink if started too early

## Progress Log

### 2026-03-08

- Added `speechgr/encoders/mimi/` with a minimal `MimiEncoder` scaffold that supports injected dummy tokenizers for local smoke tests and lazy external loading for real Mimi checkpoints.
- Registered the `mimi` encoder and added SLUE-SQA5 Mimi config scaffolding in `configs/data/slue_sqa5_mimi.yaml` and `configs/prepare/slue_sqa5_mimi.yaml`.
- Added a CPU-only smoke test covering Mimi registry wiring, audio encoding, cache writing, and cache loading.

## Implementation Notes / Open Issues

- The exact external Mimi dependency and checkpoint contract are still not pinned in-repo. The encoder currently expects either an injected tokenizer for tests or a real `model_name_or_path` for later integration.
- `discrete_code_num: 2048` and `special_token: 2048` are scaffolding defaults. We still need to confirm Mimi vocabulary size and whether multi-codebook outputs should be flattened, offset, or packed differently for retrieval.
- The new data config records the plan assumptions for `30s` passages, `15s` stride, and `3-8s` query spans, but the current retrieval/precompute pipeline does not yet materialize those span boundaries from raw SLUE audio.

## Open Questions

- exact `cluster + leaf` split for v1: `1+2` or `1+3`
- exact passage embedding source for DocID building
- exact threshold for switching the default backbone to decoder-only

## References That Matter For This Plan

- RIPOR for relevance-based, coarse-to-fine semantic DocIDs
- NAIL for the fact that learned indices can be implemented with decoder-only or encoder-decoder families
- `Exploring Training and Inference Scaling Laws in Generative Retrieval` for why decoder-only remains important later
- `Generative Retrieval as Multi-Vector Dense Retrieval` as the caution against adding complexity before proving the core gain
