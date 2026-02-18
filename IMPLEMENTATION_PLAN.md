# Implementation Plan: S2GSR-UnitY

Based on the `S2GSR_Technical_Spec.md` and the current codebase analysis, here is the detailed implementation plan.

## 1. Phase 0: Neural Indexer (RQ-VAE Upgrade)

**Goal:** Implement `DocumentRQVAE` which pools variable-length audio into a single latent vector before quantization.

### 1.1 Update `speechgr/models/rqvae.py`
- **Add `AttentiveStatisticsPooling` class:**
  - Input: `(B, T, D)`
  - Learnable weights to compute attention scores.
  - Output: `(B, D)` (Weighted sum).
- **Add `DocumentRQVAE` class:**
  - Wraps the existing `ResidualVectorQuantizer`.
  - Flow: `Input -> Encoder -> Pooling -> ResidualVQ -> Decoder`.
  - Returns: `codes` of shape `(B, num_quantizers)` (one tuple per document).

### 1.2 Update `speechgr/encoders/rqvae/encoder.py`
- Update `RQVAEEncoder` to support the new `DocumentRQVAE`.
- Add logic to handle `encode_audio` returning a single code tuple instead of a sequence of codes.
- Update `precompute` to save these document-level codes.

## 2. Phase 1 & 2: S2GSR-UnitY Model

**Goal:** Implement the "UnitY" architecture with two decoders and cross-attention fusion.

### 2.1 Create `speechgr/models/unity.py`
- **Class `UnitySpeechModel`**:
  - **Encoder**: Wrapper around `WavLM` (using `SSLModelWrapper` or similar) + `Conv1d` (stride 2).
  - **Decoder 1 (Semantic)**: Standard `T5Stack` or `TransformerDecoder`.
    - Predicts Semantic IDs (Vocabulary 5000).
  - **Decoder 2 (Retrieval)**: Standard `T5Stack` or `TransformerDecoder`.
    - **Fusion**: Custom Cross-Attention block that concatenates Encoder outputs ($H_{enc}$) and Decoder 1 hidden states ($H_{dec1}$).
    - Predicts RQ Codes (Vocabulary 256).

### 2.2 Model Configuration
- Define `UnityConfig` to handle parameters for both decoders and the fusion strategy.

## 3. Data Loading (Dual-Task)

**Goal:** Implement the mixed-batch strategy for Indexing and Retrieval.

### 3.1 Create `speechgr/data/dual_task.py`
- **Class `DualTaskDataset`**:
  - Wrapper around `SlueSQA5DatasetV2` (or base dataset).
  - **Method `__getitem__`**:
    - With probability 0.5: **Task A (Indexing)**
      - Load full document audio.
      - Target: Document ID + Semantic ID.
    - With probability 0.5: **Task B (Retrieval)**
      - Randomly crop audio (e.g., 3 seconds).
      - Target: Document ID + Semantic ID.
- **Collator**:
  - Handle padding for audio and targets.
  - Ensure batches contain mixed tasks or alternate batches (if preferred).

## 4. Inference (Trie-Constrained)

**Goal:** Enforce valid RQ code generation during beam search.

### 4.1 Create `speechgr/utils/trie.py`
- **Class `Trie`**:
  - `add(sequence)`: Insert a valid code sequence.
  - `get_children(prefix)`: Return valid next tokens.
- **Class `TrieLogitsProcessor`** (extends `transformers.LogitsProcessor`):
  - Uses `Trie` to mask invalid logits during `model.generate`.

## 5. Training Script

### 5.1 Create `scripts/train_unity.py`
- Initialize `UnitySpeechModel`.
- Initialize `DualTaskDataset`.
- Setup Loss: $\mathcal{L}_{CE}(Dec1) + \lambda \mathcal{L}_{CE}(Dec2)$.
- Training loop with mixed batches.

---

## Progress Tracking (TODO)

### 1. Phase 0: Neural Indexer (RQ-VAE Upgrade) - [DONE]
- [x] Implement `AttentiveStatisticsPooling` in `speechgr/models/rqvae.py`.
- [x] Implement `DocumentRQVAE` for fixed-length document indexing.
- [x] Update `RQVAEEncoder` to support document-level codes.
- [x] Create `scripts/phase0_prep/generate_indices.py` for automated indexing.
- [x] **Update:** Enhanced RQ-VAE architecture with `ResBlock` layers for better codebook usage. **Note:** Requires retraining RQ-VAE.

### 2. Phase 1 & 2: S2GSR-UnitY Model - [DONE]
- [x] Create `speechgr/models/unity.py` with `UnitySpeechModel` (Dual Decoders + Fusion).
- [x] Implement UnitY Fusion logic (concat acoustic + semantic hidden states).

### 3. Data Loading & Training - [DONE]
- [x] Create `speechgr/data/dual_task.py` with `DualTaskDataset` and `DualTaskCollator`.
- [x] Implement Mixed-Batch Strategy (Indexing vs. Retrieval tasks).
- [x] Create `scripts/train_unity.py` for dual-loss optimization.

### 4. Inference & Validation - [DONE]
- [x] Create `speechgr/utils/trie.py` for constrained decoding.
- [x] Implement `TrieLogitsProcessor`.
- [x] Verify pipeline with `scripts/test/smoke_test_unity.py`.
- [x] Implement Beam Search in `UnitySpeechModel.generate` using `transformers.BeamSearchScorer`.
- [x] Implement Recall@K evaluation metrics in `scripts/phase2_eval/evaluate_unity.py`.

### 5. Experiments & Scaling - [PENDING]
- [ ] **Step 1 (Phase 0):** Generate indices for LibriSpeech-100.
  ```bash
  uv run python scripts/phase0_prep/generate_indices.py \
      --audio_root /path/to/librispeech/train-clean-100 \
      --output_dir outputs/indices/ls100_v1 \
      --train_kmeans --kmeans_k 500
  ```
- [ ] **Step 2 (Phase 1):** Pre-train on LibriSpeech.
  ```bash
  uv run python scripts/phase1_train/train_unity.py \
      --id_map outputs/indices/ls100_v1/id_map.json \
      --semantic_map outputs/indices/ls100_v1/semantic_map.json \
      --audio_root /path/to/librispeech/train-clean-100
  ```
- [ ] **Step 3 (Phase 2):** Fine-tune on SLUE-SQA5.
- [ ] **Step 4:** Run ablation studies (Fusion effect, Trie constraint).
