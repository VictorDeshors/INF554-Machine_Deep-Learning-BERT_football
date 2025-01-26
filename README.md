# Kaggle INF554 Machine and Deep Learning

## Victor DESHORS, Rodrigue REIBEL, Adrien BINDEL

# Goal of the project : 
We are provided a Twitter dataset centered on the 2010 and 2014 FIFA World Cup tournaments, organized by one-minute long 
time periods. Each period is annotated with a binary label: 0 if no notable event occurred, or 1 if
a significant event—such as a goal, half time, kick-off, full time, penalty, red card, yellow card, or
own goal—occurred within that period. For an interval to be labeled as containing an event, it
must align closely with the actual event time, without excessive delay.

We built two models to predict the occurrence of notable events within specified one-minute intervals of a football match.

# Report :
A report is available in [Report.pdf](Report.pdf). I also included our oral presentation for the project in [Presentation.pdf](Presentation.pdf).

## 1. Structure of the code

- *src/*
    - contains class and function definitions for data manipulation, training, tuning and evaluation.

- *scripts/*
    - *create_BERT_df.py* : creates the df of BERT embeddings from a given df of tweets. You can modify the number of tweets per minute to embedd by modifying N_tweets
    - *add_new_features.py* : add new features and does a PCA on the embedding features.
    - *eval* folder : evaluation on eval dataset
    - *tuning* folder : parameter tuning for a given model (cf. [Report.pdf](Report.pdf))

- *challenge_data/*
    - contains folders *train* and *eval*
    - *train* contains :
        - *train_BERT_PCA/* after new features were added and PCA done on embeding features
        - *train_BERT/* for BERT embeddings once *create_BERT_df.py* is run
        - *train_tweets/* with csv files of the original data
    - *eval* contains : 
        - *eval_BERT_PCA/* after new features were added and PCA done on embeding features
        - *eval_BERT/* for BERT embeddings once *create_BERT_df.py* is run
        - *eval_tweets/* with csv files of the original data

- *models/*
    - stores trained DL models (of type CNNBinaryClassifier)

- *submissions/*
    - stores submission files for kaggle

## 2. Code usage

0. **Python Environment** :
    - We advise to use packages that are indicated in requirements.txt.

1. **Original data** :
    - The data is initially located in zip files in folders *challenge_data/train/train_tweets/* and *challenge_data/eval/eval_tweets/*.
    - You need to extract the data and let it in *challenge_data/train/train_tweets/* and *challenge_data/eval/eval_tweets/*.

2. **First create BERT embeddings for train and eval dataset** :
    - run `python -m scripts.create_BERT_df train`
    - run `python -m scripts.create_BERT_df eval`

3. **Add new features and do a PCA of the BERT embedings** :
    - run `python -m scripts.add_new_features train`
    - run `python -m scripts.add_new_features eval`

4. **Then evaluate the models**:
- *Cluster - classifier model*
    - run `python -m scripts.eval.cluster_classifier_eval`
- *CNN binary classifier model*
    - run `python -m scripts.eval.CNN_classifier_eval`

5. **Modifying the models** :
If you want to fine-tune the models, you can modify the parameters in the .py files and run : 

- *Cluster - classifier model*
    - run `python scripts.tuning.cluster_classifier_tuning`
- *CNN binary classifier model*
    - run `python scripts/tuning/CNN_classifier_tuning.py`

