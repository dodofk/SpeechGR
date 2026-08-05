# Review rubric

The draft is "okay" only when all hard gates pass and no major concern remains.

## Hard gates

1. **Evidence provenance**
   - Every quantitative claim is present in `results_ledger.md`.
   - Thesis results, the new BM25 reproduction, and imported prior work are
     visibly distinguished.
   - No post-thesis neural-model result is presented as thesis evidence.
   - Imported prior-work results are labelled as reported rather than
     reproduced.
2. **Protocol honesty**
   - Evaluation splits are named only where the thesis establishes them.
   - The paper does not invent seeds, confidence intervals, hyperparameters, or
     baseline details.
   - Known thesis inconsistencies are corrected or disclosed.
3. **Claim precision**
   - `index-free` means no external document-embedding index or ANN search.
   - The valid-DocID trie is described as constrained decoding, not as a dense
     retrieval index.
   - The paper does not claim state of the art, "fully textless," a verified
     unwritten-language result, or priority without evidence.
4. **Submission compliance**
   - Official NeurIPS 2026 `dblblindworkshop` style is used unchanged.
   - The submission is anonymous and contains no identifying paths or metadata.
   - Content ends by page 8; references begin afterward.

## Quality gates

1. The introduction has a clear problem--gap--approach--evidence chain and
   exactly three contributions.
2. The method explains the two training paths and why passage identifiers can
   be generated from acoustic units.
3. The main comparison makes transcript/KD/index differences visible.
   Candidate-pool mismatches are explicit.
4. Every negative ablation states a hypothesis, exact observed change, and a
   local lesson without universal causal language.
5. The paper uses no more than one overview figure and three result tables.
6. The abstract is one paragraph, approximately 150--180 words, and contains
   the verified headline values.
7. The rendered PDF has no clipped or overlapping content, unreadable tables,
   broken references, placeholder text, or material overfull boxes.
8. A skeptical reviewer can identify the contribution and the principal
   limitation from the first page alone.
