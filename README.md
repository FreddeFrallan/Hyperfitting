# Hyperfitting: Sharpening and Stabilizing LLMs for Open-Ended Text Generation

This repository contains the official implementation and supplementary materials for the paper **"The Hyperfitting Phenomenon: Sharpening and Stabilizing LLMs for Open-Ended Text Generation"**.

## Overview

Hyperfitting is a counter-intuitive process where pre-trained large language models (LLMs) are fine-tuned to near-zero training loss (aggressively overfit) on very small datasets. Despite poor validation loss, this drastically improves long-sequence generation, particularly with greedy decoding. This leads to:
- Highly sharp and stable modeling space where top-ranked tokens are assigned very high probabilities.
- Reduced repetition and increased diversity.
- Human-preferred outputs over traditional sampling methods.

The hyperfitting phenomenon extends to LLMs of various sizes, different domains, and even autoregressive image generation. We further find this phenomena to be distinctly different from that of Grokking and double descent.
<!--
Figure 1 Placeholder
-->
*Caption: Example of greedy decoding using Llama 3.1 and its hyperfitted counterpart. Color indicates how repetitive the generated text is. *

## Experiments conducted:
- **Human Preference Collection**: Over 20,000 annotations showed hyperfitted models were preferred for long-sequence generation, even when compared to larger models and more sophishisticated sampling techniques.
- **Sharpened Predictions**: Hyperfitting reduced entropy, collapsing predictions to favor top-ranked tokens.  
- **Data Shuffling**: Training on shuffled datasets (with the same content) resulted in ~30% different top-1 predictions, highlighting stochasticity.
- **Training Data Quantity**: Tests reducing the number of training samples were conducted, with good results as low as 8 samples (batch size). 
- **Citation Blocking**: Blocking repeated subsequences had minimal impact on output quality.
- **Cross-Modality Generalization**: Hyperfitting improved autoregressive image generation, reducing repetition in image generation.
- **Downstream Task Performance**: Hyperfitted models were evaluated on GLUE and MMLU. Hyperfitting marginally negatively impacted performance.

## Quickstart
### Requirements
- Dependencies listed in `requirements.txt`.

### Installation
1. Clone the repository:

<!--
PLACEHOLDER FOR CLONING INSTRUCTIONS
-->
   
Install dependencies:
pip install -r requirements.txt

Running Experiments
Fine-tune a pre-trained model:

<!--
PLACEHOLDER FOR HOW TO RUN EXPERIMENTS AND EVALUATE
-->

<!--
Table 1 Placeholder
-->
Caption: Human preference and type-token ratio comparison for baseline vs. hyperfitted models.

<!--
Figure 6 Placeholder
-->

Caption: Generated images showing decreased repetition.

Citation

If you use this work, please cite:

@article{carlsson2024hyperfitting,
  title={The Hyperfitting Phenomenon: Sharpening and Stabilizing LLMs for Open-Ended Text Generation},
  author={Fredrik Carlsson and Fangyu Liu and Daniel Ward and Murathan Kurfali and Joakim Nivre},
  journal={arXiv preprint arXiv:2412.04318},
  year={2024}
}
For detailed analysis, refer to the paper.
