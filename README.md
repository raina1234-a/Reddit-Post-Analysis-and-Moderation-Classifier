# Reddit Post Analysis and Moderation Classifier

An AI-powered Reddit post analysis and moderation project combining traditional Machine Learning, Natural Language Processing (NLP), and Large Language Models (LLMs) to analyze post content, identify potentially NSFW material, and estimate moderation-removal risk.

## Overview

Online communities generate large amounts of user content that may require moderation. This project explores how machine learning models and LLM-based classification can assist in analyzing Reddit posts.

The system combines TF-IDF-based text classification, metadata-driven machine learning, and Groq-hosted LLM inference to produce structured content classifications and explanations.

## Features

* **Text preprocessing:** Clean and prepare Reddit post titles for NLP analysis.
* **TF-IDF vectorization:** Convert textual content into numerical features for machine learning.
* **NSFW classification:** Analyze whether a post may contain adult content using Reddit's `over_18` flag as a reference label.
* **Removal-risk analysis:** Explore the likelihood of moderation removal using a separate machine learning task.
* **XGBoost classification:** Analyze relationships between post metadata and moderation outcomes.
* **LLM-powered moderation:** Integrate Groq-hosted language models to classify content and provide explanations.
* **Structured JSON output:** Return a content label, confidence score, removal-risk category, and reasoning.
* **Model evaluation:** Evaluate classification performance using precision, recall, F1-score, and confusion matrices.

## System Architecture

1. **Data collection:** Load Reddit post data into a Pandas DataFrame.
2. **Preprocessing:** Handle missing values and prepare text and metadata features.
3. **Feature extraction:** Apply TF-IDF vectorization to post titles.
4. **Machine learning:** Train and evaluate traditional classifiers.
5. **LLM integration:** Send selected post text to a Groq-hosted model for structured classification.
6. **Evaluation:** Compare predictions against available dataset labels.
7. **Output:** Generate interpretable JSON results for individual posts.

## Technologies Used

* Python
* Pandas and NumPy
* Scikit-learn
* TF-IDF
* XGBoost
* Natural Language Processing (NLP)
* Groq API
* Large Language Models (LLMs)
* Google Colab
* Git and GitHub

## Model Evaluation

The project evaluates model performance using metrics suited to imbalanced classification tasks.

| Metric           | Purpose                                                 |
| ---------------- | ------------------------------------------------------- |
| Precision        | Measures how many predicted positive cases are correct. |
| Recall           | Measures how many actual positive cases are identified. |
| F1-score         | Balances precision and recall.                          |
| Confusion matrix | Shows correct and incorrect predictions by class.       |

**Reported experimental result:** XGBoost deletion-likelihood classification achieved an F1-score of **0.72** only if supported by a separately verified experiment. The currently documented runs in this project produced lower F1-scores, so the 0.72 result should not be treated as reproduced until the corresponding experiment and evaluation are available.

The `over_18` column is a platform-provided adult-content flag, not a perfect ground-truth NSFW label. Similarly, removal-risk predictions are estimates rather than actual Reddit moderation decisions.

## Example Output

The Groq classifier returns structured results in the following format:

```json
{
  "content_label": "SAFE",
  "confidence": 0.95,
  "removal_risk": "LOW",
  "reason": "No clear indication of adult content."
}
```

*Example output for illustration; actual model predictions may differ.*

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/reddit-moderation-classifier.git
cd reddit-moderation-classifier
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost groq
```

### 3. Configure the Groq API key

For Google Colab, add a secret named `GROQ_API_KEY` and enable notebook access.

```python
from google.colab import userdata
from groq import Groq

client = Groq(
    api_key=userdata.get("GROQ_API_KEY")
)
```

Never commit API keys or other credentials to GitHub.

### 4. Run the project

Open the notebook in Google Colab, configure the API key, load the dataset, and execute the preprocessing, training, inference, and evaluation cells.

## Limitations

* Dataset labels may not perfectly represent actual NSFW content or moderation decisions.
* Class imbalance can significantly affect precision and recall.
* LLM predictions and confidence scores are not guaranteed to be correct or calibrated.
* Post-publication engagement features, such as score and comment count, may introduce information that would not be available when a post is first submitted.
* API usage is subject to the model's availability and account-specific rate limits.

## Future Improvements

* Evaluate the system on a larger, representative, manually reviewed test set.
* Improve minority-class detection and threshold selection.
* Compare TF-IDF classifiers with transformer-based text models.
* Develop a hybrid ML–LLM decision pipeline.
* Add error analysis, model monitoring, and explainability.
* Separate real-time content moderation from post-publication removal-risk analysis.
