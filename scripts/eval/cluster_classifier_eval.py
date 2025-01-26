"""
We run our cluster-classifier model on the evaluation dataset.
"""
#%% Imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pandas as pd
from src.ClusterClassifier import ClusterEstimator
from src import utils
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

pd.options.display.float_format = '{:.2f}'.format  # Limit to 2 decimal places

if __name__ == "__main__":

    ##########################
    ### READ TRAINING DATA ###
    ##########################
    path_dfs = "challenge_data/train/train_BERT_PCA/"
    df_BERT = utils.read_BERT(path_dfs=path_dfs)
    X, y = utils.df_to_tensors(df_BERT)

    #####################
    ### FIT THE MODEL ###
    #####################
    clf = MLPClassifier(**{'solver': 'adam', 'max_iter': 1000, 'learning_rate': 'adaptive',
                        'hidden_layer_sizes': (100,), 'alpha': 0.01, 'activation': 'tanh'})

    cluster_clf = ClusterEstimator(n_clusters=1, min_cluster_size=5, cluster_selection_epsilon=0.07,
                                cluster_selection_method="eom", alpha=1.0, clf=clf)

    cluster_clf.fit(X, y)

    ##################################
    ### PREDICT ON EVALUATION DATA ###
    ##################################
    path_eval = "challenge_data/eval/eval_BERT_PCA/"
    path_write = "submissions/cluster_clf.csv"

    df_eval = utils.read_BERT(path_dfs=path_eval)
    X_eval, ids = utils.df_to_tensors(df_eval, eval=True)

    y_eval = cluster_clf.predict(X_eval)
    df_kaggle = pd.DataFrame(data= {"ID": ids, "EventType": y_eval})
    df_kaggle.to_csv(path_write, index=False)

    ###################################
    ### EVALUATION ON TRAINING DATA ###
    ###################################
    y_pred = cluster_clf.predict(X)

    #########################
    ### CALCULATE METRICS ###
    #########################
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')

    #####################
    ### PRINT RESULTS ###
    #####################
    print('\n', "Classification Report on Training Data:")
    print(classification_report(y, y_pred))
    print("\nModel Performance Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}", '\n')

    print("No Evaluation possible on test data, as we do not have the ground truth labels.")