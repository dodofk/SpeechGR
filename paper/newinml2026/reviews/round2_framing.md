# Round 2 framing review

## Verdict

**OK.**

The framing now supports the strongest claim available from thesis-only
evidence: SpeechGR is a feasibility study of spoken-query generative retrieval
on English SLUE-SQA-5, accompanied by scoped design hypotheses rather than
causal or broad superiority claims.

The abstract contains **152 words**, is a single paragraph, and includes the
verified 5.81 Hit@1 and 18.452 Hit@20 values. The Introduction contains
**exactly three contribution items**. Page 1 makes both the contribution and
the principal limitation visible: the abstract states the spoken-query
DocID-generation contribution and explicitly limits the evidence to one
English benchmark, unresolved ablation protocols, and missing repeated-run
uncertainty.

The title accurately signals the evidential level. The Introduction has a
clear problem--gap--approach--evidence chain. Its novelty claim is also
appropriately bounded: it contrasts text-query cross-modal GR and
spoken-query dense retrieval with spoken-query DocID generation, while making
no priority claim. The updated Related Work includes the relevant 2025
spoken-input retrieval neighbors, WavRAG and VoxRAG; both use embedding-based
retrieval, so they do not contradict the stated design-point distinction.

## Hard-gate failures

**None.**

- All quantitative paper claims are represented in the thesis result ledger;
  the ranking-loss row now reports the thesis's approximate changes rather than
  reconstructed absolute results.
- Imported SpeechDPR results are labelled as reported and not reproduced. Its
  exact prior-work split and supervision mismatch are visible in the main-table
  caption.
- Unknown ablation splits, shared settings, baseline definitions,
  hyperparameters, and the codebook-data inconsistency are disclosed rather
  than invented.
- `Index-free` is consistently defined as no external document-embedding
  index or ANN search. The valid-DocID trie is consistently described as
  constrained decoding.
- The paper makes no state-of-the-art, priority, fully-textless, or verified
  unwritten-language claim.
- The official `dblblindworkshop` style is active, author and PDF metadata are
  anonymous, content ends on page 7, and references follow.

## Major concerns

**None.**

The earlier major framing problems have been resolved: the principal
limitation is concrete on page 1; “controlled/identify/matter” language has
been replaced by associational wording; the three contribution bullets are
distinct; current spoken-audio retrieval neighbors are positioned; repeated
audit caveats have been consolidated; cross-references resolve; and the figure
now separates training targets from inference-time constrained decoding.

## Minor suggestions

These are polish items and do not block the **OK** verdict.

1. In the Introduction, “Most speech-retrieval systems either ...” is broader
   than the two cited examples establish. “Representative speech-retrieval
   systems either ...” would be maximally conservative.
2. “No benchmark transcripts” is accurate, but “no SLUE-SQA-5 query or passage
   transcripts” would remove any momentary ambiguity with Flan-T5's text
   pretraining. The method and limitation already make this distinction
   correctly.
3. In Limitations, prefer “The thesis does not report repeated runs,
   confidence intervals, or significance tests” over “were not recorded,”
   which could imply knowledge about artifacts outside the thesis.
4. Figure 1 is now semantically correct and readable. The fine-tuning and
   inference group boxes remain visually crowded where the target-DocID node
   meets the inference lane; a few millimeters of extra horizontal spacing
   would improve polish, but there is no text collision or loss of legibility.
5. The raw Semantic Scholar URLs and mixed bibliographic venue formatting are
   visually inelegant. Replacing them with canonical DOI/venue entries would
   improve publication finish without changing the framing.
