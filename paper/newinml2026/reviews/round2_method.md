# Round 2 method and reproducibility review

## Verdict: NOT OK

All four rubric hard gates pass, and the scientific wording is substantially
improved. However, one major visual concern remains: Table 1 and its caption
run outside the normal text block to the left edge of page 5. Under the rubric,
“OK” requires zero major concerns, so this artifact is not yet ready.

## Hard-gate failures

**None.**

- **Thesis-only evidence:** All reported values are represented in the result
  ledger; the ranking-loss row now reports deltas rather than reconstructed
  absolute scores; SpeechDPR values are clearly imported and not reproduced.
- **Protocol honesty:** Ablation splits and shared settings are not invented.
  The 100h/960h codebook conflict and missing reproduction details are
  disclosed in rendered prose.
- **Claim precision:** `index-free` is consistently defined as no external
  document-embedding index or ANN search. The trie is accurately described as
  a valid-output-prefix constraint. The paper avoids state-of-the-art,
  priority, fully-textless, and unwritten-language claims.
- **Submission compliance:** The official
  `\usepackage[dblblindworkshop]{neurips_2026}` style is used. The PDF metadata
  has no author identity, the visible author block is the official anonymous
  review block, and no identifying local paths are visible. Content ends on
  page 7 and references begin immediately afterward, continuing through page
  9; therefore the content is within the eight-page limit and the extra pages
  are references.

## Major concerns

### 1. Table 1 violates the normal text margins

On page 5, Table 1 begins at the physical left edge of the rendered page, and
its caption spans essentially the full paper width rather than the normal
NeurIPS text block. This is visually inconsistent with Table 2, Table 3, and
the body text. It looks clipped/off-center even though the values remain
readable. This fails the polished-layout quality gate and is the only remaining
major blocker.

The likely cause is the use of `table*` plus `tabularx{\linewidth}` in this
single-column workshop layout. Change the main table to a normal `table`
environment constrained to the body `\linewidth`, then re-render. If the
six-column table does not fit, shorten the text rather than shrinking the font
or changing margins. A compact version can use:

```latex
\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\linewidth}{@{}l l >{\raggedright\arraybackslash}X l rr@{}}
...
\end{tabularx}
\caption{...}
\end{table}
```

Possible safe abbreviations are `task transcripts` for the SpeechGR training
signal and `params. + DocID trie` for its retrieval store. The rebuilt table
and every caption line must stay inside the same left/right boundaries as the
body text.

## Minor suggestions

1. **Separate the figure group boxes cleanly.** Figure 1 is now semantically
   correct: fine-tuning targets are distinct from inference outputs, and the
   trie feeds valid-next-token constraints to the decoder. However, the
   `fine-tuning` and `inference` background rectangles overlap slightly around
   the target-DocID and inference-query nodes. Move the inference group a little
   right, reduce the target box width, or reduce `inner sep` so the group
   boundaries have a visible gap. This is a small visual issue, not a method
   error.

2. **Make the input notation fully consistent.** In the task formulation,
   `\(P_\theta(y\mid q)\)` could be written
   `\(P_\theta(y\mid\mathbf{u}(q))\)` because the model receives discrete units,
   not the waveform directly.

3. **Keep the trie wording exactly as it is.** The current prose, figure, and
   caption correctly distinguish an identifier-prefix trie from an embedding
   index. The Table 1 entry `parameters + DocID trie` is also transparent about
   what is retained at retrieval time.

4. **The reproducibility disclosure is sufficient for this evidence-limited
   conversion, but remains a scientific limitation.** The missing truncation,
   sampling, optimization, and completed-beam scoring details prevent exact
   reproduction; the paper now says so plainly in Limitations. Do not restore
   confident implementation details unless they are recovered from evidence
   allowed by the thesis-only policy.

5. **Clean internal TODO comments before packaging source.** Their uncertainty
   is now represented in the rendered paper, so the comments no longer hide a
   protocol issue. Removing or converting them to neutral source notes would
   make an eventual source release look finished.

6. **Final visual pass after the table fix:** confirm no undefined references
   or material overfull boxes, inspect all nine rendered pages, and verify that
   content still ends no later than page 8. The current build has no broken
   references or visible hyperlink boxes.
