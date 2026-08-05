# Round 1 method and reproducibility review

## Verdict

**Major revision; not yet “okay.”** The conceptual method is understandable,
the two training paths are present, and the paper uses `index-free` and
`transcript-free` more carefully than the thesis. However, the rendered paper
currently fails the no-broken-reference/no-unreadable-figure gate, the overview
figure is semantically wrong about the trie, and several details required to
reproduce the method remain only as non-rendered TODO comments. The paper also
makes two claims more strongly than the thesis record permits: that BPE
“consistently” degrades retrieval and that the shared acoustic front end
“learns” jointly with Flan-T5.

The thesis-only evidence gate otherwise passes: I found no post-thesis result,
and imported SpeechDPR values are labelled as reported rather than reproduced.
The precise definition of `index-free` also passes.

## Hard-gate failures

1. **Rendered-PDF quality gate fails.**
   - The current PDF shows `Section ??` twice, in the BPE paragraph and after
     the CE objective.
   - Figure 1 has overlapping arrow labels (`indexing`, `cross-entropy`, and
     `masks invalid next tokens`) and is not publication-legible.
   - The Table 3 caption runs essentially full bleed across the page.
   - The PDF is stale relative to the source: `main.pdf` was built at 15:26:10,
     while `experiments.tex` was modified at 15:26:56. The missing
     `sec:analysis` label is present in the current source, so rebuilding to
     convergence should fix the `??`, but the rebuilt PDF must be inspected
     again.

2. **Figure correctness / claim-precision gate fails.**
   Figure 1 draws the inference-only trie as feeding a box labelled
   `gold DocID`. At inference there is no gold DocID; the trie masks the
   decoder's next-token distribution before a generated DocID is completed.
   The figure also places training and inference in one box, making the dashed
   trie arrow appear to modify the training target. This contradicts the
   otherwise-correct prose.

3. **Protocol-honesty gate fails for a known inconsistency.**
   The 100-hour versus 960-hour LibriSpeech codebook-training discrepancy
   appears only in `% TODO` comments. Comments are neither disclosure nor part
   of the reviewed paper. This must be resolved from the original artifact or
   disclosed to the reader.

4. **Submission-quality gate fails while critical method metadata remain
   unresolved.**
   The current source does not report maximum input length/truncation,
   indexing-versus-retrieval sampling, task markers, optimizer, learning rate,
   batch size, fine-tuning duration, model-selection rule, or completed-beam
   scoring. These are material for acoustic sequences and variable-length
   string DocIDs. If the values cannot be recovered, their absence must be
   disclosed explicitly; leaving them in comments is unacceptable.

## Major concerns, ranked

### 1. The retrieval-training protocol is not reproducible

Sections 3.4 and 4.1 say the two paths are trained jointly, but do not say how
the examples are combined. A reader cannot determine whether every passage is
seen once per epoch, whether query examples dominate, whether batches mix both
tasks, or whether task prefixes distinguish passage and query inputs. Because
passages and queries have substantially different lengths, this choice can
materially affect the result.

After the source audit, replace the first part of Section 3.4 with:

```latex
We form an indexing set
\(\mathcal{D}_{\mathrm{idx}}=\{(\mathbf{u}(p_i),y_i)\}_{i=1}^{N}\)
and a retrieval set
\(\mathcal{D}_{\mathrm{ret}}=\{(\mathbf{u}(q_j),y_{t(j)})\}_{j=1}^{M}\),
where \(t(j)\) maps query \(j\) to its gold passage. During fine-tuning,
we [AUDITED SAMPLING RULE, INCLUDING THE INDEXING:RETRIEVAL RATIO].
[AUDITED TASK-PREFIX OR “NO TASK PREFIX IS USED.”] Both paths update the
same Flan-T5 parameters and use the cross-entropy objective below.
```

Do not say that the acoustic front end “learns” from both paths unless HuBERT
and the codebook were actually updated. If they were fixed preprocessing, use:

```latex
Both paths use the same fixed discrete-unit preprocessing and update the
shared Flan-T5 encoder--decoder parameters.
```

Add one compact setup sentence containing the audited optimization details:

```latex
Inputs are truncated/padded to [AUDITED LENGTH AND POLICY]. We optimize with
[OPTIMIZER] at learning rate [LR], batch size [BATCH DEFINITION], for
[STEPS/EPOCHS], and select the checkpoint by [METRIC] on [SPLIT].
```

If any field is genuinely unrecoverable, do not leave a TODO. Add to
Limitations:

```latex
The archived experiment record does not preserve [EXACT MISSING FIELDS], so
the reported system cannot be exactly reproduced from the paper alone.
```

### 2. Figure 1 must be redrawn as three distinct stages

Use three visually separated panels or rows:

1. **Shared fixed preprocessing:** `waveform -> HuBERT -> k-means ->
   deduplicate -> u(a)`.
2. **Fine-tuning:** two arrows,
   `u(p_i) -> shared Flan-T5 -> target y_i` and
   `u(q_j) -> shared Flan-T5 -> target y_{t(j)}`, with CE shown beside the
   targets.
3. **Inference:** `u(q) -> Flan-T5 decoder -> masked next-token scores ->
   generated DocID`, with a side arrow from `valid-DocID trie` to
   `masked next-token scores`.

Delete the `gold DocID` label from the inference path. The trie arrow must
point to the decoder scores, not to the output identifier. Put `indexing` and
`retrieval` at the left of their rows rather than on top of arrows, and remove
the enclosing label `retrieval training and decoding`, which conflates phases.

Recommended caption:

```latex
\caption{\textbf{SpeechGR overview.} A shared HuBERT--\(k\)-means
preprocessor maps passages and queries to deduplicated acoustic units.
Fine-tuning combines passage-to-DocID indexing examples and
spoken-query-to-DocID retrieval examples. At inference, a trie masks invalid
next-token continuations during beam search; it stores valid output prefixes,
not document embeddings.}
```

### 3. Identifier construction and decoding scores are underspecified

The method says corpus strings are tokenized and that atomic identifiers are
“one consecutive integer token.” It does not establish whether atomic IDs are
new single vocabulary entries or decimal strings that may tokenize into
multiple pieces. Calling them single-token is only valid in the former case.
The paper also omits an example of the corpus DocID, the resulting token
lengths, and how variable-length completed beams are scored. Those details are
especially important because the main argument for atomic IDs concerns
multi-token decoding errors.

After auditing the implementation, rewrite the identifier paragraph as:

```latex
For the main system, passage \(p_i\) uses the corpus identifier
\texttt{[AUDITED FORMAT/EXAMPLE]}, which the Flan-T5 tokenizer maps to
[AUDITED TOKENIZATION DESCRIPTION]. For the atomic ablation, we
[“add one new output-vocabulary entry per passage” OR THE ACTUAL
IMPLEMENTATION]. We call this identifier single-token only when one added
entry represents the complete passage ID.
```

Rewrite constrained retrieval to expose the actual ranking rule:

```latex
At each step, the trie masks tokens that do not extend a valid tokenized
DocID. We decode with beam width 20 and rank completed identifiers by
[AUDITED SUM/AVERAGE/LENGTH-NORMALIZED LOG-PROBABILITY], using
[AUDITED EOS AND STOPPING RULE]. Each completed identifier maps uniquely to
one corpus passage.
```

### 4. Two method statements exceed the evidence

- `BPE ... consistently degrades retrieval` is too broad. The paper has
  matched no-BPE controls for \(K=500\) and \(K=1000\), but no \(K=2000\)
  no-BPE control, and the pretraining metadata for granularity rows are
  unresolved.
- `the same acoustic front end and encoder--decoder parameters learn` implies
  HuBERT/codebook optimization, which the thesis does not establish.

Exact replacements:

```latex
BPE is evaluated as an optional compression ablation and is not used in the
final \(K=500\) configuration; Section~\ref{sec:analysis} reports the two
matched comparisons available in the experiment record.
```

and, subject to confirming the front end was frozen:

```latex
Both paths use the same discrete-unit preprocessing and update the shared
Flan-T5 encoder--decoder.
```

In the abstract, `BPE compression ... do not improve the tested baseline` is
acceptable because it is scoped to the tested setting. Avoid `consistently`
and any universal causal interpretation elsewhere.

### 5. The manuscript should be self-contained rather than narrating a thesis audit

Phrases such as “the thesis explicitly labels,” “transcribed from the thesis,”
and “until the original evaluation scripts are audited” expose the drafting
process and ask reviewers to reason about an unavailable source. They may also
make a public thesis easier to connect to an anonymous submission. Preserve
the uncertainty, but state it as a property of the reported experiment record.

Exact replacements:

```latex
The headline result uses the reported test split. The available records do not
identify the evaluation split for each ablation configuration, so we report
those rows as split-unresolved.
```

```latex
\caption{Core design study. Values are percentages. The evaluation split is
not recorded per ablation row, and the 10-epoch value equals the headline test
result. The records also do not identify which granularity runs omit
speech-unit pretraining.}
```

```latex
Repeated runs, confidence intervals, and statistical significance were not
recorded; small differences should therefore be interpreted cautiously.
```

## Minor concerns

1. **Task notation can be tighter.** The formulation defines the corpus but not
   the query set or the query-to-passage mapping. Use
   \(\mathcal{Q}=\{(q_j,t(j))\}_{j=1}^{M}\) and target \(y_{t(j)}\) throughout,
   including Figure 1. This eliminates the current mix of \(y_i\) and \(y^+\).
2. **Scope `transcript-free` once and reuse it.** Recommended sentence:

   ```latex
   SpeechGR's acoustic-unit adaptation, retrieval fine-tuning, and inference
   do not consume query or passage transcripts. Flan-T5 is nevertheless
   text-pretrained, and the DocIDs are metadata-derived strings.
   ```

   The current pretraining interpretation is otherwise good: it correctly
   says LibriLight span corruption adapts the model to acoustic units and does
   not teach the target Spoken-Wikipedia-to-DocID mapping.
3. **Avoid asserting which parameters pretraining updates until audited.**
   Replace “adapts the embedding table and transformer” with “adapts the
   expanded encoder--decoder to discrete speech units” unless parameter
   freezing is recovered.
4. **Codebook provenance must be reader-visible.** Resolve 100h versus 960h.
   If not recoverable, say “The archived configuration does not identify
   whether the \(k\)-means codebook was fitted on the 100h or 960h LibriSpeech
   subset.” This is inferior to resolving it but honest.
5. **Link borders are visually noisy.** If permitted by the official template,
   use `\usepackage[hidelinks]{hyperref}` or an equivalent non-identifying
   link style. Do not modify `neurips_2026.sty`.
6. **Rebuild and inspect the exact submission artifact.** Compile to
   convergence, confirm zero undefined references and material overfull boxes,
   and visually inspect all eight pages. The standard anonymous affiliation
   block comes from the official review style and should not be “fixed” by
   editing the style file.

## TODO triage before the next review

**Must resolve or disclose in rendered prose:** codebook training data;
identifier/tokenizer construction; indexing/retrieval sampling; task prefixes;
maximum input length and truncation; optimizer; learning rate; batch size;
fine-tuning duration; model selection; front-end freezing; beam scoring,
normalization, EOS/stopping; evaluation split per ablation if recoverable.

**May be disclosed as unavailable in Limitations:** random seeds, number of
runs, confidence intervals, hardware, exact span-length sampling, and detailed
unused-token-to-acoustic-token initialization mapping.

**Already handled adequately:** no claim that LibriLight pretraining
internalizes the target corpus; no state-of-the-art claim; text-pretrained
Flan-T5 caveat; SpeechDPR values marked as imported; trie correctly defined in
the prose as a decoding constraint rather than a dense index.
