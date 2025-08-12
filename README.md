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

The dataset is to be located in the `/data` subfolder, file names `real.csv` and `fake.csv`. But the dataset is too large (>100MB) for github so it is not included in the Github repository.

At the end I have also added a manual test dataset with 5 fake and 5 real articles with strategically placed cues. It is present in `/data` directory too.

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
│   ├── real.csv                   # Dataset of real news articles - Not in Github repository (file too large)
│   ├── fake.csv                   # Dataset of fake news articles - Not in Github repository (file too large)
│   ├── X_train_bert.npy           # BERT embeddings for the train split of the data - Not in Github repository (file too large)
│   ├── X_test_bert.npy            # BERT embeddings for the test split of the data - Not in Github repository (file too large)
│   └── manual_test.csv            # Additional dataset for manual test
├── images/                        # Directory for storing visualization images
├── credibility-checker.ipynb      # Jupyter Notebook for all code to train models
├── requirement.txt               # Python dependencies to be installed to run the project
└── README.md                      # Project documentation and analysis results
```

## Usage
1. Clone the repository:
    - Open terminal and run:
      `git clone https://github.com/nachiketpatil/credibility-checker.git`
2. Install dependencies:
   `pip install -r requirement.txt`
3. Open the Jupyter Notebook:
   `jupyter notebook credibility-checker.ipynb`
   Run the cells to explore the analysis.

## License
This project is public (Unlicense). Feel free to download and use the code for your own analysis.

--------------------------------------------------------------------------------------------------------------------

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

**Confusion Matrix and ROC plot for best performaing models:**

1.Confusion Matrix and ROC plot for Random Forest model:
  <img width="1200" height="600" alt="RandomForest_tfidf_conf_matrix_roc_plot" src="https://github.com/user-attachments/assets/34f197a6-b381-4105-8d46-8da8a69b31db" />

2. Confusion Matrix and ROC plot for Decision Tree model:
   <img width="1200" height="600" alt="Decision_Tree_tfidf_conf_matrix_roc_plot" src="https://github.com/user-attachments/assets/74f26e39-140f-41c6-9522-5891b686182d" />



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



### Tuning the Simple ML models with GridSearch CV to find best hyperparameters: 
In line with our goal to find the best model that classifies the text articles as real or fake, let us improve on the simple models by tuning the Hyperparameters with GridSearchCV:

Following are the performance metrics for running GridSearch Cross Validation to tune the models:
| Model              | Train Time (s) | Test Accuracy | Train Accuracy | Recall Score | F1 Score |
|--------------------|----------------|---------------|----------------|--------------|----------|
| LogisticRegression | 11.5885        | 0.9849        | 1.0000         | 0.9868       | 0.9838   |
| DecisionTree       | 28.9275        | 0.9941        | 0.9968         | 0.9978       | 0.9936   |
| KNN                | 317.5199       | 0.6694        | 1.0000         | 0.3290       | 0.4804   |
| SVM                | 10912.0330     | 0.9739        | 0.9999         | 0.9692       | 0.9719   |
| RandomForest       | 129.2147       | 0.9957        | 1.0000         | 0.9986       | 0.9954   |

Compared to running standalone models, we see minor improvements in the accuracy andrecall for all the models after Hyperparameter tuning.

**More inferences:**
* Overall high performance for most models – Logistic Regression, Decision Tree, SVM, and Random Forest all achieved test accuracies above 97%, indicating strong predictive capability on the dataset.

* Random Forest had best recall (0.9986) and also posted near-perfect train and test accuracies, showing strong sensitivity to positive cases but with a possible risk of overfitting given its perfect training accuracy.

* Decision Tree achieved the highest test accuracy (0.9941) with very high recall (0.9978), performing nearly as well as Random Forest with less computational time.

* SVM delivered strong performance (test accuracy 0.9739, recall 0.9692) but required by far the longest training time (over 10,000 seconds), making it less practical for large-scale or rapid retraining scenarios.

* KNN underperformed significantly (test accuracy 0.6694, recall 0.3290) compared to other models, despite perfect training accuracy, suggesting severe overfitting and poor generalization.

* Logistic Regression offered an excellent trade-off – very high test accuracy (0.9849), recall (0.9868), and extremely fast training time (≈12 seconds), making it a highly efficient baseline model.


#### Confusion Matrices and ROC plots for tuned best models:
1. Confusion Matrix and ROC plot for tuned Random Forest model:
  <img width="1200" height="600" alt="Random Forest Classification Model with Tuned Hyperparameters_conf_matrix_roc_plot" src="https://github.com/user-attachments/assets/ac529c36-48de-4916-a73f-a77cf8c7902d" />


2. Confusion Matrix and ROC plot for Decision Tree model:
   <img width="1200" height="600" alt="Decision Tree Model with Tuned Hyperparameters_conf_matrix_roc_plot" src="https://github.com/user-attachments/assets/e4cf6630-a3b6-4690-8c06-0435f9302b26" />



### Plotting the feature importance of the best models:
On top of performance metrics and confusion matrix showing hits and misses, best way to visualize what affects the real or fake decision is plotting the features that affect the classification decision most. Here our features are words in the news article text. Following are the feature importance plots for some of the most accurate simple models we have tested:

1. Feature importance for top 50 features for Random forest model:
   <img width="1200" height="600" alt="random_forest_tuned_feature_influence" src="https://github.com/user-attachments/assets/2054b846-2add-4152-86be-60f18fe19a53" />

2. Feature importance for top 50 features for Decision Tree model:
   <img width="1200" height="600" alt="decision_tree_tuned_feature_influence" src="https://github.com/user-attachments/assets/3b602e0a-5052-4b1d-a02d-1d8d32ac24f5" />


3. Feature importance for top 50 features for Logistic Regression model:
   <img width="1200" height="600" alt="logistic_regression_tuned_feature_influence" src="https://github.com/user-attachments/assets/d59744f8-6844-453b-bdcd-7626ddd8282e" />


### Observations, Limitations and Solutions:
**Observation:** Based on fearure importance charts above, we can see that the classification decistion is influenced by inividual word tokens. For any given context, these models perform really well like news articles with majority of news along the subjects of politics and world events i.e. these models are overfit on the fake-real-news dataset. But if these models are trained with fictional or fantasy novels, the results will vary wildly for real news. We will test this scenario later.

**Limitation:** In real languages, words individually do not affect the truth or fakeness of the articles. The words used in context with some combination of other words - sentenses, semantics matter. Above traditional simple models although perform well in any given scenario, they have severe limitations in case of Natural Language processing. In short these traditional ML models with TF-IDF rely on sparse, handcrafted features that ignore word order, syntax, and contextual meaning. They struggle with capturing semantic relationships and contextual meaning. 

**Solutions:** Deep learning methods such as Convolutional Neural Networks (CNN),  Recursive neural network (Long-Short Term Memory - LSTM) and Transformers based model (Bidirectional Encoder Representations from Transformers - BERT) overcome the limitations of traditional ML models by learning dense, context-aware representations directly from raw text.

### Training and evaluating Newral networks on the real - fake news dataset:

#### Convolutional Neural Network (CNN):
* CNN work on the data in multiple layers of filters where each layer converts the data into the classification decision step by step.
* Raw text is tokenized into integers optionally and then in embedding layer it is mapped to dense vectors.
* Then most important is convolutional layer where filters slide over word sequence windows of fixed length (kernel size). This makes sure the words are taken in context of nearby words as context. We are using **kernel size of 5**.
* The Dense Layer takes the output of previous layers and introduces non-lineaity with ReLU (Rectified Linear Unit) activation so the network can learn complex relationships.
* Last layer is Sigmoid activation for binary classification → probability that text is real/fake.


Following is the model training history and confusion matrix of the CNN: ***<<<Update>>>***
<img width="1200" height="500" alt="CNN_history_accuracy_loss" src="https://github.com/user-attachments/assets/0ff4e4ed-c9da-4a97-927a-837cdd7cc747" />

<img width="600" height="500" alt="CNN_confustion_matrix" src="https://github.com/user-attachments/assets/a7fec6a9-444a-4b23-a1cf-76ce7c108492" />

Here we ran the CNN for 10 epochs same for following RNN. We can observe that the train and test accuracies converge at **epoch 2**. After that the accuracy gain or reduction in loss is not significant.
Also from the coinfusion matrix, we can observe that the false negatives and false positives are significantly reduced compared to the simple ML models like Random Forest classifier or decision tree model.

**Shortcomings?**
Although CNNs are fash to train and very good at local pattern detection (key phrases, sentiment indicators), they have some shortcomings.
* As CNNs run with fixed small kernel size, which is the sliding window of consecutive words, they only see local context within the kernel size (e.g., 3–5 words at a time)
* In the model, max pooling layer loses exact positional information as the sliding window boundary is crossed or the words are too far apart for the Dense layer to establish the relationship — model knows what pattern exists, but not where or in what sequence.
* Normally CNNs are good at handling fixed context length tasks.

There are other advanced models that overcome these shortcomings.
* Recursive Neural Network models like LSTM can process sequences word by word, carrying hidden states forward, so they can theoretically remember important information from far earlier in the text.
* LSTM processes sequentially, inherently preserving order and temporal structure, which is critical for grammar, meaning, and sentiment flow.
* LSTM is not bound by contexct length. The layers carry over the context so it is good at processing whole articles


#### Recursive Neural Netweork - Long-Short Term Memory Model:
Long-Short Term Memory (LSTM) model was selected for this news articles dataset classification as it is one of the best and simple newral network models available.
LSTM is one of the Recusrsive neural network models where the accuracy is improved and loss is reduces for each run of the model as the LSTM layer learns the word patterns during each epoch. 

* LSTM captures long term word dependencies and remembers full context irrespective of the length of the sequence. This overcones some drawbacks of the CNN before.
* Most important layer in LSTM is the Long Short-Term Memory layer (RNN variant): It reads the embedded sequence word-by-word, keeping track of context through hidden states. Captures order and long-term dependencies in the text (important for meaning). The final output (after the last word) is passed on as a captured summary of the whole sequence.
* Then the Dense Layer takes the output of previous layers and introduces non-lineaity with ReLU (Rectified Linear Unit) activation so the network can learn complex relationships.


Following is the model training history and confusion matrix of the LSTM: ***<<<Update>>>***
<img width="1200" height="500" alt="LSTM_history_accuracy_loss" src="https://github.com/user-attachments/assets/e0d6ed2f-0065-4590-a2be-3f60aa3052bb" />

<img width="600" height="500" alt="LSTM_confustion_matrix" src="https://github.com/user-attachments/assets/911bbbbb-9841-42c0-972b-00fad2a3efe1" />


#### Comparing the CNN and LSTM performance:
| Model | Train Time (s) | Test Accuracy | Train Accuracy | Recall Score | F1 Score |
|-------|----------------|---------------|----------------|--------------|----------|
| CNN   | 124.65         | 0.998546      | 0.999608       | 0.998314     | 0.998434 |
| LSTM  | 798.92         | 0.995077      | 0.998573       | 0.992775     | 0.994691 |


* Both models achieved extremely high accuracy — CNN at 99.85% and LSTM at 99.51% on the test set.
* Training accuracy is near perfect for both (above 99.85%), showing they fit the training data very well.
* Recall and F1-scores are also exceptionally high for both models, indicating strong and balanced performance.
* CNN trained much faster (~125 seconds) compared to LSTM (~799 seconds), making it more computationally efficient for similar accuracy.
* Overall, CNN offers comparable performance to LSTM but with significantly lower training time, making it the more efficient choice for this task.


### Manual Testing:
From the exercise of training multiple models and observing their pros and cons, I have crafted some articles that exploit these pros and cons of these various models.

Lets run the manual testing dataset of real nad fake news articles through these models and observe their classification.

* This manual_test dataset has 10 articles.
* 5 real like (based on real news but not real) and 5 fake articles.
* The real articles have most of the telltale signs of the news articles (e.g. real sources like Reuter / NPR, professional language and phrases.)
* The Fake articles also have the telltale signs like no soure included, unprofessional language like extreme exclamations and publicly available image sources.


Following is the accuracy chart for all previous models on the manual testing data:
| Model              | Test Accuracy | Recall Score | F1 Score  | False Positives | False Negatives |
|--------------------|---------------|--------------|-----------|-----------------|-----------------|
| LogisticRegression | 0.70          | 0.40         | 0.571429  | 0               | 3               |
| DecisionTree       | 0.70          | 0.40         | 0.571429  | 0               | 3               |
| KNN                | 0.40          | 0.00         | 0.000000  | 1               | 5               |
| SVM                | 0.70          | 0.60         | 0.666667  | 1               | 2               |
| RandomForest       | 0.70          | 0.40         | 0.571429  | 0               | 3               |
| CNN                | 0.70          | 0.40         | 0.571429  | 0               | 3               |
| LSTM               | 0.80          | 0.60         | 0.750000  | 0               | 2               |



<p align="center">
  <img alt="manual_test_LogisticRegression_confustion_matrix" src="https://github.com/user-attachments/assets/0d548b57-2148-4bc4-b05d-e84fe8b799ee" width="45%"/>
  <img alt="manual_test_DecisionTree_confustion_matrix" src="https://github.com/user-attachments/assets/38f3ef2b-fd64-4100-b9a1-36e1bbbc134c" width="45%"/>
</p>

<p align="center">
  <img alt="manual_test_K-Nearest Neighbors_confustion_matrix" src="https://github.com/user-attachments/assets/dccbf185-1ca4-4521-a345-6764f08665bb" width="45%" />
  <img alt="manual_test_Support Vector Machine_confustion_matrix" src="https://github.com/user-attachments/assets/000dab77-a21f-4238-82c2-cc2c0529bf09" width="45%" />
</p>

<img alt="manual_test_Random Forest Machine_confustion_matrix" src="https://github.com/user-attachments/assets/0ef31112-e18f-4776-bf0c-d4e2fc65a81f" />

<p align="center">
  <img alt="manual_test_Convolutional Neural Network_confustion_matrix" src="https://github.com/user-attachments/assets/c8e8a97d-6771-4a6a-91dc-30e4565efe60" width="45%" />
  <img alt="manual_test_Long- Short Term Memory (RNN)_confustion_matrix" src="https://github.com/user-attachments/assets/d27ff93a-1eb4-4705-b4bd-1f7ee98c1d42" width="45%" />
</p>


#### Observations:
For the manual testing of the random 10 articles with 5 real and 5 fake, most of the models classified the articles along the lines of the previous results of test accuracies on the large real - fake news dataset.
* K-Nearest neighbors model performed the worst with lowest accuracy, recall and F1 scores and highest false negatives. 
* SVM also predicted one false positive and 2 false Negatives with low accuracy, recall and F1 scores.
* For the manual testing dataset, Logistic Regression, Decision Trees and Random forst behaved same with same accuracy, recall, f1 scores and no false positives but 3 false negatives.
* CNN also predicted no false positive but found 3 dalse Negatives. Some of the phrases or words placed in the text were far apart which indicate the article to be true. CNN Missed them for 3 articles
* LSTM found the far placed text or rather absense of the phrases in articles at least once more. It performed the best for manual test dataset Indicating it can carry some context throughout the article.
* Most of the clear indicators of the new article being fake (e.g. no soure included, unprofessional language like extreme exclamations and publicly available image sources) were caught by almost all models.


### Final Model comparison and Overall conclusion:
After extensive testing — from baseline Naive Bayes to tuned Random Forest, SVM, and deep learning models (CNN and LSTM) — several trends were observed:

For the dataset with larger text to be classified (1. real - fake news dataset and 2. manual testing dataset):
* Deep learning models like Convoluted Neural Networks and Recursive Neural Network like LSTM worked best with extremely high accuracy (≥99.5%) and CNN taking significantly shorter duration to train. At the same time LSTM detected contextually distant cues better than CNN on the manual testing dataset.
* Tuned Random Forest and Decision Tree also performed near-perfect (≈99.4–99.5% accuracy), with strong recall and F1 scores. These Tree-based models are quite fast to train but, show reduced performance outside the training distribution.
* Logistic Regression was the fastest traditional ML model with accuracy close to 98.5%.
* KNN consistently underperformed due to high-dimensionality issues with TF-IDF and poor generalization.
* SVM performed admirably similar to Logistic Regression and Decision Tree models but was excruciatingly slow to train again because of high-dimentionaliity of the data.

* There are other transformer models, like BERT, available to be run on the news dataset but are quite difficult to train and tuning them is out of scope of this project.


### Business Recommendations:
For large text articles classification:
* For best contextual accuracy and for nuanced and long-form news, LSTM is preferred, as it retains and uses long-range dependencies in text.
* If speed and scalability are the primary concern, CNN is the most practical choice, delivering near-LSTM accuracy with much lower training time.
* For a lightweight baseline, Logistic Regression with TF-IDF offers a fast, resource-efficient option with competitive performance.
* Avoid KNN and SVM for any Natural Language Processing as they do not handle high dimentionality of the large articles.

* While LSTM works best for larger texts, CNN might performa best for shorter text articles like shorter social media posts and non-binary classifications like sentimental analysis.


### Future work:
**Advanced models:**
In this project, we trained from baseline model (Naive Bayes), Simple Classification models (Logistic Regression, Decision Trees, K-Nearest Neighbors, Support Vector Machine), Ensemble model (Random Forest) and couple of simple Neural Network models (Convolutional Neural Network, Recursive Neural Network with Long-Short Term Memory model). 

There are several advanced models like Bi-directional LSTM, complex RNNs and Transformer-based models (BERT, RoBERTa, or DistilBERT).

**Fine-tune the advanced models**
Pre-trained BERT was used for text vectorization but fine tuning it was out of scope for this project.
Fine-tuning these advanced models on this dataset could bridge the contextual gap and improve performance on subtle or adversarial fake news examples.

**Incorporate metadata and multimodal signals**
Use article source credibility, publication date, author history, and even associated images for richer classification.


| Contact Information | |
|-------|---------|
| **Name** | Nachiket Patil |
| **Email** | <hi.nachiket.patil@gmail.com> |
| **GitHub** | [https://github.com/nachiketpatil) |
| **LinkedIn** | [https://www.linkedin.com/in/nachikethiralalpatil/) |
| **Project Repository** | [https://github.com/nachiketpatil/credibility-checker.git) |
| **Primary Data Source** | [https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/data) |

