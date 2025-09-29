# Abstractive Text Summarizer using Transformers

This project focuses on building and fine-tuning a Transformer-based model to generate concise, abstractive summaries of news articles. The project involved training models on two different datasets and navigating various technical challenges to achieve strong results. The final models are available on the Hugging Face Hub.

## Project Overview

The main objective was to develop an effective abstractive text summarizer. Unlike extractive methods that copy sentences, abstractive summarization generates new sentences that capture the core meaning of the source text, similar to how a human would summarize a document.

This project documents the entire process from model selection and data preparation to fine-tuning and evaluation, using the Hugging Face ecosystem on the Kaggle platform.

## Models

Two models were fine-tuned using the **`facebook/bart-base`** architecture, which is well-suited for text generation tasks. Each model was trained on a different standard summarization dataset:

1. **souradeepdutta/bart-base-summarizer**: Fine-tuned on the **CNN/DailyMail** dataset. This dataset consists of news articles and human-written bullet-point highlights.
2. **souradeepdutta/bart-base-summarizer-xsum**: Fine-tuned on the **XSUM (Extreme Summarization)** dataset. The summaries in this dataset are single, highly abstractive sentences.

## Technology Stack

* **Platform**: Kaggle Notebooks with NVIDIA T4 GPU hardware.
* **Core Libraries**:

  * Hugging Face `transformers` for model architecture and training
  * Hugging Face `datasets` for data loading and processing
  * Hugging Face `evaluate` for performance metrics
  * PyTorch as the deep learning framework
  * NLTK for text processing during evaluation

## Setup and Usage

You can use the fine-tuned models directly from the Hugging Face Hub for summarization tasks.

### 1. Installation

First, install the necessary libraries.

```bash
pip install transformers torch
```

### 2. Summarization Pipeline

Use the `pipeline` API from `transformers` for a straightforward way to generate summaries.

```python
from transformers import pipeline

# Load the model fine-tuned on the XSUM dataset
summarizer = pipeline("summarization", model="souradeepdutta/bart-base-summarizer-xsum")

# Example article
test_article = """
NASA's James Webb Space Telescope has captured its first direct image of a planet outside our solar system.
The exoplanet, known as HIP 65426 b, is a gas giant about six to 12 times the mass of Jupiter.
This observation is a transformative moment for astronomy, as it points the way toward future observations
that will reveal more information than ever before about exoplanets. The telescope's advanced infrared
capabilities allow it to see past the glare of the host star to capture the faint planet.
"""

# Generate summary
summary = summarizer(
    test_article,
    max_length=80,
    min_length=15,
    do_sample=False
)

print(summary[0]['summary_text'])
```

## Results

The models were evaluated using the **ROUGE** metric, which measures the overlap between the generated summary and a human-written reference summary.

### Quantitative Results

The final scores on a test set of 5,000 examples are as follows:

| Model/Dataset                  | ROUGE-1 | ROUGE-2 | ROUGE-L |
| :----------------------------- | :-----: | :-----: | :-----: |
| **BART-base on CNN/DailyMail** |  39.75  |  17.29  |  26.94  |
| **BART-base on XSUM**          |  38.98  |  16.21  |  31.45  |

*Table: Final ROUGE scores for the fine-tuned models*

### Qualitative Analysis

A key finding was the difference in behavior between the two models, largely due to the nature of their training data. The model trained on CNN/DailyMail exhibited **"lead bias,"** where it learned to copy the first few sentences of an article, as news articles often start with the most important information.

The XSUM model, trained on more abstractive single-sentence summaries, produced more genuine and concise summaries.

**Example Comparison:**

* **Sample Article**: "NASA's James Webb Space Telescope has captured its first direct image of a planet outside our solar system. The exoplanet, known as HIP 65426 b, is a gas giant about six to 12 times the mass of Jupiter..."
* **Summary from CNN/DailyMail Model**: "NASA's James Webb Space Telescope has captured its first direct image of a planet outside our solar system. The exoplanet is a gas giant about six to 12 times the mass of Jupiter. This is a transformative moment for astronomy." (More extractive)
* **Summary from XSUM Model**: "Astronomers have used the James Webb Space Telescope to take the first direct image of an exoplanet, a gas giant known as HIP 65426 b." (More abstractive)

