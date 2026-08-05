# Round 6 literature review: query-modality gap

Verdict: **OK** after revision.

- The broad claim that multimodal GR only accepts text queries was rejected.
  IRGen accepts image queries, and GENIUS accepts text, image, and image--text
  queries.
- GRACE, CART, and GTA accurately support the narrower observation that many
  cross-modal generative retrievers broaden the target modality while retaining
  natural-language queries.
- The manuscript now claims only that direct spoken-query identifier generation
  remains largely unexplored. It makes no categorical first-work claim.
- WavRAG is correctly treated as dense spoken-query retrieval, not a prior
  autoregressive DocID generator.

