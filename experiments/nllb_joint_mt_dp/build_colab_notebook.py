#!/usr/bin/env python3
"""Build the independent Colab launcher for the NLLB two-head experiment."""

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "notebooks" / "nllb_joint_mt_dp_colab.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip() + "\n"}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


cells = [
    markdown(
        """
        # NLLB two-head Mandarin→Cantonese translation and dependency parsing

        This is a separate experiment from the Wu-style transition model. A pinned pretrained
        `facebook/nllb-200-distilled-600M` decoder supplies two outputs:

        1. its language-model head generates Cantonese UD tokens;
        2. a biaffine graph head predicts HEAD and DEPREL over the same decoder states.

        Mandarin dependency trees are not used. Non-projective target trees need no SWAP because
        the parser scores all possible head–dependent arcs directly. Model selection uses DEV only;
        TEST is evaluated after loading one frozen jointly selected checkpoint.

        Use a GPU runtime, then choose **Runtime → Run all**. The final cell downloads
        `nllb_joint_mt_dp_results.zip`. NLLB is CC BY-NC 4.0 and intended here for research.
        """
    ),
    markdown(
        """
        ## 1. Install the pinned runtime

        PyTorch is supplied by Colab. The remaining versions are pinned so the custom training loop,
        PEFT adapter checkpoint, tokenizer, metrics, and non-projective MST decoder stay reproducible.
        """
    ),
    code(
        r"""
        %pip -q install "transformers==4.46.3" "peft==0.13.2" "accelerate==1.1.1" "safetensors==0.4.5" "sentencepiece==0.2.0" "sacrebleu==2.5.1" "networkx==3.4.2" "pandas==2.2.3" "matplotlib==3.9.2" "tqdm==4.67.1"
        """
    ),
    markdown(
        """
        ## 2. Fetch the independent experiment source

        The notebook launcher and Python implementation live in separate files. This keeps the new
        pretrained two-head model isolated from the Wu notebook and makes the actual training code
        reviewable outside notebook JSON.
        """
    ),
    code(
        r"""
        import subprocess
        from pathlib import Path

        repo = Path("/content/zh-yue-d2d")
        if repo.exists():
            subprocess.run(["git", "-C", str(repo), "pull", "--ff-only"], check=True)
        else:
            subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ss-sebastian/zh-yue-d2d.git", str(repo)], check=True)
        %cd /content/zh-yue-d2d/experiments/nllb_joint_mt_dp
        """
    ),
    markdown(
        """
        ## 3. Run the complete experiment

        This downloads checksum-pinned UD r2.18 data, resolves all 1,004 pairs through explicit
        `parallel_id`, creates the fixed 803/100/101 split, downloads the pinned NLLB revision,
        trains LoRA plus the biaffine head, selects one checkpoint on the predeclared joint DEV score,
        evaluates TEST once, shows a simultaneous sentence/tree example, and downloads every artifact.

        An existing atomic vertical-bar token is used only as a word-boundary control token. It is
        removed before text output. Boundary counts are asserted for every supervised target so a
        subword/UD-node alignment error stops the run instead of silently corrupting HEAD labels.
        """
    ),
    code(
        r"""
        %run /content/zh-yue-d2d/experiments/nllb_joint_mt_dp/run_colab.py
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "nllb_joint_mt_dp_colab.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("Wrote", OUTPUT)
