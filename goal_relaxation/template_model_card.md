---
datasets:
- sentence-transformers/sentence-compression
language:
- en
metrics:
- sari
- rouge
base_model:
- facebook/bart-large
pipeline_tag: text-generation
tags:
- sentence-compression
- sentence-simplification
---

## Fine-Tuned BART-Large for Sentence Compression

### Model Overview

This model is a fine-tuned version of ```facebook/bart-large``` trained on the ```sentence-transformers/sentence-compression``` dataset. The goal of this model is to generate compressed versions of input sentences while maintaining fluency and meaning.

### Training Details

Base Model: ```facebook/bart-large```

Dataset: ```sentence-transformers/sentence-compression```

Batch Size: 8

Epochs: 5

Learning Rate: 2e-5

Weight Decay: 0.01

Evaluation Metric for Best Model: SARI Penalized

Precision Mode: FP16 for efficient training

### Evaluation Results

### Validation Set Performance:

SARI: {sari_valid}

SARI Penalized: {sari_penalized_valid}

ROUGE-1: {rouge1_valid}

ROUGE-2: {rouge2_valid}

ROUGE-L: {rougel_valid}

### Test Set Performance:

SARI: {sari_test}

SARI Penalized: {sari_penalized_test}

ROUGE-1: {rouge1_test}

ROUGE-2: {rouge2_test}

ROUGE-L: {rougel_test}

### Training Loss Curve

The loss curves during training are visualized in bart-large-sentence-compression_loss.eps, showing both training and evaluation loss over steps.


