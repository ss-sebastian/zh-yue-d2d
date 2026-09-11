# NLLB joint Mandarin→Cantonese translation and dependency parsing

This experiment is independent of the Wu-style transition model. It does not import or modify the Wu notebook.

The model starts from pinned `facebook/nllb-200-distilled-600M` weights and has two heads over one decoder:

1. NLLB's existing language-model head generates Cantonese UD tokens.
2. A new biaffine graph head predicts one HEAD and DEPREL per generated UD token.

The Mandarin dependency tree is intentionally not used. Target non-projective trees are learned directly as graphs, so there is no SHIFT/REDUCE/SWAP oracle. A pre-existing atomic vertical-bar vocabulary item is used only as an internal UD-word separator; generated user-facing output never contains it. No new randomly initialized output-vocabulary row is required.

Open `notebooks/nllb_joint_mt_dp_colab.ipynb` in Colab and run all cells. NLLB is licensed CC BY-NC 4.0 and is intended here for research, not unreviewed production or commercial deployment.

Decoding is greedy (`num_beams=1`) and has no length penalty; beam search and length-aware reranking are intentionally outside this first experiment.

The two losses are trained jointly. At inference, autoregressive translation necessarily establishes the target node sequence first; the graph head then scores all arcs over those decoder states. The public `translate_and_parse(source_text)` function returns the sentence and tree together and requires no Mandarin parse.
