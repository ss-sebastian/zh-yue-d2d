# Parallel dependency-distribution analysis

Primary scope: the 803 custom Cantonese train sentences and their Chinese-HK
translations matched by the official unique `parallel_id`.  No model was loaded
or modified.  The full-corpus files are descriptive appendices only and must not
be used for architecture or hyperparameter selection.

Key train-only measurements:

- Tokens: Cantonese 11,001; Chinese-HK 7,854.
- Mean non-root arc distance: Cantonese 3.400; Chinese-HK 2.833.
- Arcs of distance >=6 (denominator: all integer-ID tokens): Cantonese 14.22%; Chinese-HK 10.64%.
- Strict DEPREL JS divergence: 0.0382 bits.
- CoNLL18 base-DEPREL JS divergence: 0.0287 bits.
- Conditional attachment-signature JS divergence: 0.0977 bits.

The attachment signature is `(dependent UPOS, head UPOS, whether the head is to
the left/right, distance bucket, base DEPREL)`.  Its larger divergence shows why
marginal label frequencies alone do not determine `P(HEAD, relation | sentence)`.

Limitations: Chinese-HK is not the documented training corpus of the Mandarin
ELECTRA parser; translations are not word-aligned and use different tokenization;
these results therefore describe the two annotated parallel treebanks rather than
the checkpoint's learned representation.
