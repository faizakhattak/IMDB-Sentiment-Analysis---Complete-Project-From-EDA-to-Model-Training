# IMDB Sentiment Analysis - Complete Project From EDA to Model Training

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NLTK](https://img.shields.io/badge/NLTK-3.8.1-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2.2-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Project Description

This project presents a comprehensive end-to-end Natural Language Processing (NLP) pipeline for sentiment analysis on the IMDB movie review dataset. The notebook systematically walks through all four critical phases of a machine learning project: **Exploratory Data Analysis (EDA)**, **Text Preprocessing**, **Feature Engineering**, and **Model Training & Evaluation**.

## 📊 Dataset Overview

The dataset consists of **50,000 movie reviews**, perfectly balanced with:
- **25,000 positive reviews** 👍
- **25,000 negative reviews** 👎

Each review is accompanied by a sentiment label, making it an ideal benchmark dataset for binary sentiment classification tasks.

## 🏗️ Project Structure and Key Components

### Part 1: Exploratory Data Analysis (EDA)

The EDA phase provides deep insights into the dataset's characteristics:

- **Class Balance Analysis**: Confirmed a perfectly balanced dataset (50% positive, 50% negative)
- **Text Length Analysis**: Calculated review lengths in characters, words, and non-space characters
  - Average review length: **~1309 characters**
  - Average review length: **~231 words**
- **HTML Tag Detection**: Identified that **58.4%** of reviews contain HTML tags (primarily `<br />` tags), highlighting the need for HTML cleaning
- **Word Frequency Analysis**: Analyzed most common words before and after stopword removal, revealing sentiment-discriminative terms
- **Vocabulary Analysis**: Quantified unique words in positive (**75,766**) vs. negative (**72,867**) reviews
- **Correlation Analysis**: Found near-perfect correlation between different text length metrics

### Part 2: Text Preprocessing

A robust preprocessing pipeline was implemented to clean and normalize the text data:

| Step | Description |
|------|-------------|
| **HTML Tag Removal** | Stripped all HTML tags using regex |
| **Lowercasing** | Converted all text to lowercase for consistency |
| **Punctuation & Special Character Removal** | Removed URLs, emails, punctuation, and special characters |
| **Tokenization** | Split text into individual word tokens using NLTK |
| **Stopword Removal** | Removed common English stopwords plus custom movie-review-specific stopwords (43 additional terms) |
| **Stemming & Lemmatization** | Applied both techniques, with lemmatization chosen for the final pipeline due to its linguistic accuracy |

**Preprocessing Results**: Achieved a **54% reduction** in average token count per review (from 228 to 105 tokens)

### Part 3: Feature Engineering

Multiple feature representation techniques were implemented and compared:

| Feature Type | Description | Output Shape |
|--------------|-------------|--------------|
| **Bag-of-Words (BoW)** | Simple, binary, and filtered variants | 50,000 × 5,000 |
| **TF-IDF** | Standard, sublinear, and bigram variants | 50,000 × 5,000 |
| **N-grams** | 1-gram, 2-gram, 3-gram, and combined | Various (up to 50,000 × 5,000) |
| **Word2Vec** | Custom-trained embeddings | 50,000 × 100 |
| **GloVe** | Pre-trained embeddings (100-dim) | 50,000 × 100 |
| **VADER** | Lexicon-based sentiment scores | 50,000 × 4 |
| **TextBlob** | Polarity and subjectivity scores | 50,000 × 2 |
| **Custom Lexicon** | 8 custom features based on sentiment lexicons | 50,000 × 8 |

### Part 4: Model Training & Evaluation

Multiple machine learning models were trained and evaluated on the engineered features:

#### Models Implemented:
- **Logistic Regression**
- **Naive Bayes** (Multinomial & Bernoulli)
- **Support Vector Machines (SVM)**
- **Random Forest**
- **XGBoost**
- **Neural Networks** (Simple DNN)

## 📈 Key Visualizations

The project includes numerous visualizations to aid understanding:

- ✅ Sentiment distribution bar charts
- ✅ Text length histograms and box plots
- ✅ Word frequency bar charts
- ✅ Discriminative words analysis
- ✅ BoW vs. TF-IDF comparisons
- ✅ N-gram frequency analysis
- ✅ 2D projections of TF-IDF and Word2Vec features using SVD/PCA
- ✅ VADER and TextBlob score distributions
- ✅ Confusion Matrix for model performance
- ✅ ROC Curves comparing all models
- ✅ Feature Importance plots for tree-based models

## 🛠️ Installation & Requirements

```bash
# Clone the repository
git clone https://github.com/yourusername/IMDB-Sentiment-Analysis.git

# Install required packages
pip install -r requirements.txt
