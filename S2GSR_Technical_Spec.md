# Technical Specification: S2GSR-UnitY Implementation

## Version: 4.1 (Final Master Plan)
**Target:** ICASSP/INTERSPEECH
**Constraint:** Pure Speech / Zero-Resource (No Text Transcripts)

This document provides the complete technical details for implementing the S2GSR-UnitY architecture. It integrates the Deep RQ-VAE Neural Indexer, the Dual-Task Training Strategy (Indexing + Retrieval), and Trie-Constrained inference into a single execution plan.

---

## 1. System Overview

The system consists of three distinct phases:

*   **Phase 0 (Offline):** Training the Neural Indexer (RQ-VAE) and Semantic Tokenizer to generate targets ($Y_{id}, Y_{sem}$).
*   **Phase 1 (Pre-training):** Training the S2GSR-UnitY model on LibriSpeech with two simultaneous objectives: Generative Indexing (Doc $\to$ ID) and Generative Retrieval (Query $\to$ ID).
*   **Phase 2 (Fine-tuning):** Refining the model on SLUE-SQA5 with Trie-Constrained decoding.

---

## 2. Phase 0: The Neural Indexer (Offline Preparation)

**Objective:** Map every document audio to a Semantic ID (BPE) and a Retrieval ID (RQ).

### 2.1 Component A: Deep RQ-VAE (Retrieval ID)

We replace standard K-Means with a Deep Residual Quantizer to increase capacity and avoid collisions.

*   **Model:** `RQVAE` with `AttentiveStatisticsPooling`
*   **Input:** WavLM Features ($T 	imes 1024$). 
*   **Pooling:** `AttentiveStatisticsPooling`. Use learnable attention weights to aggregate $T$ frames into one $1024d$ vector.
    *   Formula: $v = ∑ 	ext{softmax}(W x_t) · x_t$
*   **Quantizer:** `ResidualVectorQuantizer` (RVQ).
    *   `dim`: 1024
    *   `num_quantizers`: 8 (Depth)
    *   `codebook_size`: 256
    *   `kmeans_init`: True
    *   `commitment_loss`: 0.25
*   **Training:**
    *   **Dataset:** LibriSpeech clean-100 + clean-360.
    *   **Loss:** MSE(Reconstruction, Original) + Commitment Loss.
*   **Output Generation:**
    *   Run inference on ALL documents (Libri + SLUE).
    *   Save `id_map.json`: `{"doc_path": [c1, c2, c3, c4, c5, c6, c7, c8]}`.

### 2.2 Component B: Semantic Tokenizer (Semantic ID)

*   **Feature Extractor:** WavLM (Layer 15).
*   **Clustering:** K-Means ($K=500$). 
*   **Sub-word Modeling:** SentencePiece (BPE) on the unit sequences.
*   **Output Generation:**
    *   Save `semantic_map.json`: `{"doc_path": [t1, t2, ... t_n]}` (Vocab Size 5000).

---

## 3. S2GSR-UnitY Model Architecture

**Objective:** The Generative Retriever.

### 3.1 Architecture Details

*   **Encoder:** WavLM-Large (Pre-trained).
    *   **Mod:** Add a `Conv1d` downsampling layer (Stride 2) after WavLM to reduce sequence length.
*   **Decoder 1 (Semantic Bridge):**
    *   Standard Transformer Decoder (4 Layers).
    *   **Cross-Attn:** Attends to Encoder outputs.
    *   **Output:** 5000-dim Softmax (BPE Tokens).
*   **Decoder 2 (Retrieval Head):**
    *   Standard Transformer Decoder (4 Layers).
    *   **Input:** Learnable `[START]` token (or previous RQ codes).
    *   **UnitY Fusion:** The Cross-Attention input is a concatenation of:
        *   $H_{enc}$ (Encoder Output)
        *   $H_{dec1}$ (Decoder 1 Last Hidden State) - This is the critical "UnitY" link.
    *   **Output:** 256-dim Softmax (RQ Codes).
    *   **Generation Strategy:** Sequential. It generates the 8 RQ codes one by one: $C_1 	o C_2 	o … 	o C_8$.

---

## 4. Phase 1: Pre-training (Dual-Task)

**Objective:** Train the model to both MEMORIZE documents and RETRIEVE them.

We split the training batch into two distinct tasks to ensure robustness.

### 4.1 Task A: Generative Indexing (Doc $\to$ DocID)

*   **Input:** Full Document Audio ($A_{doc}$). 
*   **Target:** Its own ID ($Y_{id}$) and Semantic Summary ($Y_{sem}$). 
*   **Goal:** Memorization. Ensures the model firmly links the acoustic representation of the document to its identifier.
*   **Dropout Strategy:** Apply heavier dropout (0.3) to the encoder during this task to prevent trivial identity mapping.

### 4.2 Task B: Generative Retrieval (Query $\to$ DocID)

*   **Input:** Randomly Cropped Audio ($A_{crop}$, e.g., 3 seconds).
*   **Target:** The Source Document's ID ($Y_{id}$) and Semantic Summary ($Y_{sem}$). 
*   **Goal:** Invariance. Ensures the model can map a partial, noisy view to the clean document ID (Speech-ICT).

### 4.3 Mixed-Batch Training Strategy

*   **Batch Composition:** 50% Task A (Indexing) + 50% Task B (Retrieval).
*   **Loss Function:** 
    $$ ℒ = ℒ_{Index} + ℒ_{Retrieve} $$
    Where each $ℒ$ is:
    $$ ℒ_{CE}(Dec1, Y_{sem}) + λ · ℒ_{CE}(Dec2, Y_{id}) $$
    We recommend $λ=2.0$ to prioritize the final ID prediction.

---

## 5. Phase 2: Fine-tuning (Real Queries)

**Objective:** Adapt to real question acoustics.

*   **Task:** Query $\to$ DocID.
*   **Dataset:** SLUE-SQA5.
*   **Input:** Real Question Audio.
*   **Target:** The Relevant Passage's ID ($Y_{id}$). 
*   **Replay Buffer:** Continue to include a small ratio (e.g., 10%) of Task A (Indexing) samples from LibriSpeech in the batch. This is crucial to prevent "Catastrophic Forgetting," where the model forgets the document index while learning to answer questions.

---

## 6. Phase 3: Inference (Trie-Constrained Decoding)

**Objective:** Prevent invalid ID generation.

### 6.1 Building the Trie

*   Load `id_map.json`.
*   Build a prefix tree where every path from Root $\to$ Leaf is a valid 8-code tuple from the training set.

### 6.2 Constrained Beam Search

*   **Method:** Implement a `LogitsProcessor` for HuggingFace generate.
*   **Logic:**
    ```python
    class TrieLogitsProcessor(LogitsProcessor):
        def __call__(self, input_ids, scores):
            # input_ids: The sequence generated so far (e.g., [c1, c2])
            # scores: The logits for the next token (c3)

            valid_next_tokens = self.trie.get_children(input_ids)
            mask = torch.ones_like(scores) * float('-inf')
            mask[:, valid_next_tokens] = 0

            return scores + mask
    ```
*   **Parameters:**
    *   `num_beams`: 10
    *   `max_length`: 8 (Fixed length for RQ-IDs).

---

## 7. Implementation Roadmap

1.  **Step 1 (Indexing):** Implement `AttentiveStatisticsPooling` in `speechgr/models/rqvae.py`. Train `RQVAE` on LibriSpeech. Generate `id_map.json`.
2.  **Step 2 (Tokenizer):** Train K-Means + SentencePiece. Generate `semantic_map.json`.
3.  **Step 3 (Model):** Implement `UnitySpeechModel` class in `speechgr/models/unity.py` with the specific cross-attention fusion logic.
4.  **Step 4 (Pre-train):** Implement `DualTaskDataset` that returns mixed batches of (Full Doc, ID) and (Crop, ID). Train on LibriSpeech.
5.  **Step 5 (Inference):** Implement `TrieLogitsProcessor` and evaluate Recall@10 on SLUE-SQA5.

---

## 8. Ablation Study Plan (For Paper)

To demonstrate the contribution to ICASSP reviewers, run these experiments:

*   **Dual vs. Single Task:** Compare "Retrieval-Only Training" vs "Dual Indexing+Retrieval Training". (Hypothesis: Dual training stabilizes convergence).
*   **UnitY Effect:** Remove Decoder 2's connection to Decoder 1. (Hypothesis: Performance drops significantly without the Semantic Bridge).
*   **Trie Constraint:** Turn off the Trie constraint during inference. (Hypothesis: Validity drops to <50%, Recall drops).
*   **RQ Depth:** Compare Depth=3 vs Depth=8. (Hypothesis: Depth=3 causes collisions in large corpora).
