# credibility-checker
In this project some ML models are trained to figure out what is the best model to verify credibility (classify true of fake) news articles

## Overview: 
In this practical application, we answer the question 'What is the most effective model for determining whether a news article is real or fake?'

Following CRISP-DM method.

## Motivation:
NEWS has always been a critical source of truth and information of world events that affect everybody. In an era of widespread misinformation, distinguishing real news from fake is critical to maintaining an informed public. This project is an initial foray into classifying information vs disinformation by applying machine learning to classify news credibility, offering practical insights into automated fact-checking.

## Problem Statement
The goal of this exercise is to evaluate performance and accuracy of different models in ML on the real and fake news Dataset. Find the best model to train on the datset.

## Data Description

**Source** : The data is taken from Kaggle 'fake-and-real-news-dataset' [link](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/data).

The dataset contains 2 separate CSV files one for real and one for fake news articles. The news articles are collected between years 2016 and 2017.
Both sheets contain 4 columns: Title, Text, Subject and Date.


Dataset columns:

* Title: title of news article
* Text: body text of news article
* Subject: subject of news article
* Date: publish date of news article

The dataset is taken from the UCI Machine Learning repository [link](https://archive.ics.uci.edu/ml/datasets/bank+marketing).  It comes from from a Portugese banking institution and is a collection of the results of multiple marketing campaigns. It contains client attributes like age, job, marital status, etc, there are some attributes of when was the client last contacted. Other socio-economic attributes are also included and finally the result of the marketing contact - whether the client accepted the offer of long term deposit or not.

The dataset is in the `/data` subfolder, file name `bank-additional-full.csv`.

## Dependencies
This project requires the following Python libraries:
- `matplotlib.pyplot` (for plotting)
- `seaborn` (for enhanced visualizations)
- `pandas` (for data manipulation)
- `numpy` (for numerical operations)
- `nltk` (for natural language processing)
- `sklearn`(scikit-learn for various NLP models)
- `tensorflow` (tensorflow - keras for NLP with neural networks)

To run the code either use [Jupyter Notebook](https://jupyter.org/install) or upload the notebook and directories to [Google Colab](https://colab.research.google.com/).


Local setup:
```bash
 pip install matplotlib seaborn pandas numpy nltk wordcloud tensorflow
 pip install notebook
```

## Repository Structure
```
├── data/                          # Directory for dataset
│   ├── real.csv                   # Dataset of real news articles
│   └── fake.txt                   # Dataset of fake news articles
├── images/                        # Directory for storing visualization images
├── credibility-checker.ipynb      # Jupyter Notebook for all code to train models
└── README.md                      # Project documentation and analysis results
```

## Usage
1. Clone the repository:
    - Open terminal and run:
      `git clone https://github.com/nachiketpatil/credibility-checker.git`
2. Install dependencies:
   `pip install matplotlib seaborn pandas numpy nltk wordcloud tensorflow`
3. Open the Jupyter Notebook:
   `jupyter notebook coupon_data_analysis.ipynb`
   Run the cells to explore the analysis.

## License
This project is public (Unlicense). Feel free to download and use the code for your own analysis.

## Analysis Report
### Business understanding:

### Data understanding:

#### Feature correlation to the target:

### Data Preparation:

### Baseline Models comparison:
For model comparison, we used the following classification models:
- Dummy Classifier
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors Classifier
- Support Vector Classifier

