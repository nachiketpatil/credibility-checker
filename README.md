# credibility-checker
**Author**: Nachiket Patil
In this project some ML models are trained to figure out what is the best model to verify credibility (classify true of fake) news articles

## Overview: 
In this practical application, we answer the question 'What is the most effective model for determining whether a news article is real or fake?'

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
For Natural Language Processing, the input data is language text. This data needs to be processed and converted into numerial data (vectorized input) that the models understand. Following are the steps for this process:
1. Data Collection: We have 2 separate files (real.csv and fake.csv). The contents are concatenated and shuffled to randomize the data. Then any null values and duplicates are dropped. There are no null values and no duplicate rows.
2. Text Cleaning / Preprocessing: The text column that contains main news articles is then Converted to lowercase -> Punctuations removed -> Stopwords removed (e.g., "the", "and") -> Special characters, numbers are removed -> Tokenized into separate words
3. Text Normalization: All tokenized words are then reduced to their base dictionary form (lemma) with Lemmatization. This way we have same wordswithout tense or degree. (e.g. "playing" -> "play", "luckiest" -> "lucky")
4. Vectorization: The cleaned and tokenized text is then converted into numerical format TF-IDF (Term Frequency–Inverse Document Frequency). TF-IDF assigns unique numerical value for each word based on its frequency across documents.
5. Split: Split the data into train set and test set. 80% of the rows are set for train set and remaining 20% are test set.

   Now the data is ready for training the models.


### Baseline Model:
To establish the baseline performance an daccuracy metrics for the text data, we need to run a simple and quick model. For NLP, to classify the text into binary values (real vs fake), most common baseline model is Naive Bayes Model. 

The Naive Bayes model is a simple but powerful probabilistic machine learning algorithm used for classification tasks—especially in text classification like spam detection, sentiment analysis, and fake news detection. It is based on Bayes’ Theorem, which describes the probability of a class given some evidence (features), assuming feature independence. It naively assumes that all features (words, in text) are independent of each other, which is rarely true in real-world data—but this assumption makes the model simple and fast.

The Naive Bayes model results are:
* runtime: 2.65 seconds
* accuracy: 0.9288351795904666
* precision: 0.93
* f1-score: 0.93
* recall: 0.93

<img width="640" height="480" alt="naive_bayes_confusion_matrix" src="https://github.com/user-attachments/assets/6fb57d6b-9fa3-48f0-a0a5-5d74db25b94a" />


| Contact Information | |
|-------|---------|
| **Name** | Nachiket Patil |
| **Email** | <hi.nachiket.patil@gmail.com> |
| **GitHub** | [https://github.com/nachiketpatil) |
| **LinkedIn** | [https://www.linkedin.com/in/nachikethiralalpatil/) |
| **Project Repository** | [https://github.com/nachiketpatil/credibility-checker.git) |
| **Primary Data Source** | [https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/data) |
