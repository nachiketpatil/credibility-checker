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
3. Text Normalization: All tokenized words are then reduced to their base dictionary form (lemma) with Lemmatization. This way we have same words without tense or degree. (e.g. "playing" -> "play", "luckiest" -> "lucky")
4. Vectorization: The cleaned and tokenized text is then converted into numerical format either based on term frequency (TF-IDF) or some deep learning model based contextual vectorization (BERT).
5. Split: Split the data into train set and test set. 80% of the rows are set for train set and remaining 20% are test set.


### Vectorization:
ML models (like logistic regression, SVMs, neural networks) work with numbers, not raw text or other unstructured formats. Vectorization is the process of converting non-numerical data into a numerical vector representation that algorithms can process.
We are dealing with text data, there are multiple ways to vectorize the text data. The most common methods are:
TF-IDF: TF-IDF (Term Frequency–Inverse Document Frequency) is a technique that reflects how important a word is to a document in a collection. The numerical value for a word increases with the number of times a word appears in a document (term frequency) but is offset by how common the word is across all documents (inverse document frequency).
BERT: BERT vectorization uses a pre-trained deep learning model (BERT: Bidirectional Encoder Representations from Transformers) to convert text into dense numerical embeddings that capture contextual meaning. Unlike traditional methods, BERT considers the entire sentence and the position of words to generate rich, semantic representations.

For this project, we will use both TF-IDF and BERT vectorization methods to compare the performance of different models.
TF-IDF vectorization is faster and simpler, while BERT vectorization captures needs loading a torch model and tune it for the dataset. 
BERT is far more compute intensive with vectorization time ~3 hours. So In the project, the word embeddings, output of BERT vectorization, is saved in `/data` directory and is loaded for subsequent runs.


### Baseline Model:
To establish the baseline performance an daccuracy metrics for the text data, we need to run a simple and quick model. For NLP, to classify the text into binary values (real vs fake), most common baseline model is Naive Bayes Model.

The Naive Bayes model is a simple but powerful probabilistic machine learning algorithm used for classification tasks—especially in text classification like spam detection, sentiment analysis, and fake news detection. It is based on Bayes’ Theorem, which describes the probability of a class given some evidence (features), assuming feature independence. It naively assumes that all features (words, in text) are independent of each other, which is rarely true in real-world data—but this assumption makes the model simple and fast.

The Naive Bayes model results are:
* runtime: 0.034 seconds
* train accuracy: 0.938
* test accuracy: 0.928
* precision: 0.92
* f1-score: 0.92
* recall: 0.92


<img width="1200" height="600" alt="NaiveBayes_tfidf_conf_matrix_roc_plot" src="https://github.com/user-attachments/assets/7a25fe3a-5729-494c-999c-62083ce576a2" />


### Training and comparing Simple ML models on the TF-IDF vecors:
As we are trying to determine whether the news article is real or fake, it means we are classifying the data into 2 categories. For this task, we can run simple ML models on the data and compare the performance of the models. Later we can tune hte models for optimal performance.
The models that were run: Naive Bayes, Logistic Regression, Decision Trees, K-Nerarest Neighbors, Support Vector Machine and an ensemble Random Forest model.

Following is the initial comparison table of training simple models on the news data:
| Model              | Train Time (s) | Train Accuracy | Test Accuracy | Recall Score | F1 Score |
|--------------------|----------------|----------------|---------------|--------------|----------|
| NaiveBayes         | 0.034159       | 0.938855       | 0.928619      | 0.921724     | 0.923058 |
| LogisticRegression | 0.445304       | 0.999972       | 0.984113      | 0.983141     | 0.982904 |
| DecisionTree       | 16.538965      | 0.999972       | 0.994630      | 0.994220     | 0.994220 |
| KNN                | 52.414938      | 0.588599       | 0.577982      | 0.097303     | 0.176419 |
| SVM                | 921.262353     | 0.991329       | 0.977176      | 0.979046     | 0.975522 |
| RandomForest       | 15.230548      | 0.999972       | 0.994182      | 0.994942     | 0.993745 |

Initial Inferences: 

From initial runs of the simple models, we can observe that:

* Best Performing Models: Random Forest and Decision Tree achieved the highest Test Accuracy (~99.4%) and F1 Scores (~0.995), indicating exceptional performance on both precision and recall.

* Naive Bayes: Offers very fast training (~0.07 sec) and respectable performance (F1 Score ~0.92), making it a strong lightweight option.

* Logistic Regression & SVM: High accuracy (98–98.6%), but SVM is extremely slow to train (approx 42 minutes), while Logistic Regression provides a good trade-off between performance and speed.

* K-Nearest Neighbors (KNN): Performed poorly with Test Accuracy ~56.8% and F1 Score ~0.18, likely due to high dimensionality and sparsity of TF-IDF vectors.

* Training Time vs. Performance Trade-off: Naive Bayes and Logistic Regression are efficient with solid results.

* Tree-based models like Decision Tree and Random Forest offer near-perfect accuracy but at higher computational cost.

* SVM and KNN are not optimal for large-scale TF-IDF features due to performance or speed issues.

* Another inference we can make here is Random Forest being an Ensemble of decision trees, it performs better than simple Decision Trees. The wisdom of the crowd is better than individual.


### Training and comparing Simple ML models on the BERT embeddings for the news articles:

Following the the tabular comparison of the performance metrics for simple models on BERT embeddings for the news dataset:

| Model              | Train Time (s) | Train Accuracy | Test Accuracy | Recall Score | F1 Score |
|--------------------|----------------|----------------|---------------|--------------|----------|
| LogisticRegression | 5.250392       | 0.561047       | 0.506825      | 0.378613     | 0.416314 |
| DecisionTree       | 43.709835      | 0.944030       | 0.507944      | 0.469412     | 0.469865 |
| KNN                | 8.796412       | 0.686219       | 0.500783      | 0.456888     | 0.459545 |
| SVM                | 2533.206382    | 0.537216       | 0.532670      | 0.048410     | 0.087792 |
| RandomForest       | 95.763590      | 0.944030       | 0.507160      | 0.380780     | 0.417867 |

Naive Bayes cannot be run on BERT embeddings as the embeddings contain negative numbers which Naive Bayes cannot handle negative values.

Initial Inferences:
* Overall performance is low across models:  Test accuracies are clustered around 50%, indicating the BERT embeddings did not yield strong predictive power with these simple classifiers in this setup.

* Decision Tree and Random Forest achieved high training accuracy (0.9440) but failed to generalize, with test accuracy just over 50%, suggesting significant overfitting.

* Logistic Regression and KNN showed moderate training accuracy but did not outperform chance on the test set, indicating limited separation between classes in the embedding space for these models.

* SVM performed the worst on recall and F1 (recall ≈ 0.048, F1 ≈ 0.088) despite a slightly higher test accuracy (0.5327), indicating it correctly classified very few positive cases.

* Train times varied significantly, from seconds for Logistic Regression (≈5s) and KNN (≈9s) to over 2,500 seconds for SVM, highlighting inefficiency without performance gains for this dataset.

* High training–test performance gap for tree-based methods and low recall for most models suggest the need for better regularization, feature selection, or more task-specific fine-tuning of BERT embeddings.

These results are similar to the performance of these Simple models with TF-IDF vectors. But the overall accuracy is much lower with BERT embeddings.

#### Comparing the Simple models with TF-IDF vectors vs BERT embeddings for real - fake news dataset:
The much lower accuracies (~50%) with BERT embeddings compared to TF-IDF (~97% - 99%) in our results likely come from a combination of data, feature representation, and model compatibility factors.
* BERT embeddings were not fine-tuned for this specific task. Pre-trained (torch library) BERT embeddings without task-specific fine-tuning, they do not capture the nuances of your real/fake news dataset, while TF-IDF, directly encodes term frequencies that work extremely well for linear models in text classification.
* BERT sentence/document embeddings are dense and limited to 256 or 512 maximum length which loses the remaining text in the data. While TF-IDF generates vectors for whole data.
* Mismatch between embeddings and simple ML models: BERT’s contextual embeddings shine when used with deep architectures or fine-tuned transformers which are out of the scope for this project, while Simple ML models struggle to extract useful decision boundaries from compressed semantic vectors.
* Overfitting on TF-IDF due to feature sparsity: In our performance comparisons, Decision Tree and Random Forest achieve high training and test accuracies — possibly indicate the models are overfitting on TF-IDF data. We will see how the deep learning models work better despite lower accuracies overall.








| Contact Information | |
|-------|---------|
| **Name** | Nachiket Patil |
| **Email** | <hi.nachiket.patil@gmail.com> |
| **GitHub** | [https://github.com/nachiketpatil) |
| **LinkedIn** | [https://www.linkedin.com/in/nachikethiralalpatil/) |
| **Project Repository** | [https://github.com/nachiketpatil/credibility-checker.git) |
| **Primary Data Source** | [https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/data) |




