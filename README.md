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

The dataset is in the `/data` subfolder, file names `real.csv` and `fake.csv`.

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
│   └── fake.csv                   # Dataset of fake news articles
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

## Exploratory Data Analysis
The dataset contains 2 files in comma separated values (csv) format. One for real news and one for fake news.
The dataset is in the `/data` subfolder, file names `real.csv` and `fake.csv`.
Both csv files have same structure and contain 4 columns: Title, Text, Subject and Date.

Description of the dataset columns:

* Title: title of news article
* Text: body text of news article
* Subject: subject of news article
* Date: publish date of news article

These news articles are collected between years 2016 and 2017.
This project is focusing on NLP so the date is not relevant to the classification.

Real data contains 21417 news articles.
Fake data contains 23481 new articles.
Total 44898 news articles.
<img width="600" height="600" alt="fake_vs_real_distribution" src="https://github.com/user-attachments/assets/12c51eeb-b75d-4799-a40e-b6cb294af4ae" />
The Data is balanced for real vs fake distribution. Real vs Fake percentage is almost the same (looks like the differnece is intentional).



There are 8 subjects the news are related to. But the distribution of the subjects is skewed. Mjority of the articles are related to `politics` and `world events`. Third major category is labeled `news`. This is not really relevant to training or evaluating the article text for real or fake. 
<img width="800" height="800" alt="news_distribution_by_subject" src="https://github.com/user-attachments/assets/5842bc5a-8c92-41ff-8bc9-d399d530779a" />



There are very few articles longer than 2000 words. This is relevant to lemmatization and vectorization techniques and training some of the compute heavy models like LSTM.
<img width="1000" height="600" alt="news_distribution_by_word_count" src="https://github.com/user-attachments/assets/742766e7-5a6a-4546-ac7a-bb40a3e41b3b" />



There is almost no differene between relation of word count and article credibility. i.e. just like normal word count, majority of articles whether real or fake are similarly distributed below 2000 word count.
<img width="1200" height="600" alt="fake_vs_real_distribution_by_word_count" src="https://github.com/user-attachments/assets/b9423115-0e0c-439f-9454-9bf09db79a39" />



Following are the word cloud images of the articles after preprocessing. These show that the words related to political leaders or country names are the words used most of the times in the articles.
<img width="640" height="480" alt="real_word_cloud" src="https://github.com/user-attachments/assets/6eb1d6ce-4162-460b-8349-95fe14afa760" />
<img width="640" height="480" alt="fake_word_cloud" src="https://github.com/user-attachments/assets/fb5887fc-701c-4a6e-ac74-0f0908b82e55" />


### Data Preparation:


### Baseline Models comparison:
For model comparison, we used the following classification models:
- Logistic Regression
- Support Vector Model

