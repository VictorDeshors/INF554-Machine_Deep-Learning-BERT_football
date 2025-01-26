"""
We run the CNN binary classifier model on the evaluation dataset.
"""
# %% Imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from src.TweetDataset import TweetDataset
from src.CNN.DL_utils import CNN_kaggle_eval
from src import utils
import pickle

pd.options.display.float_format = '{:.2f}'.format  # Limit to 2 decimal places
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ############################
    ### READ EVALUATION DATA ###
    ############################
    path_dfs = "challenge_data/eval/eval_BERT_PCA"
    df_BERT = utils.read_BERT(path_dfs=path_dfs)
    X_eval, ids = utils.df_to_tensors(df_BERT, eval=True)  # Convert to tensors

    ####################
    ### LOAD MODEL #####
    ####################
    path_model = "models/CNN_classifier_128.pkl"
    with open(path_model, "rb") as f:
        model = pickle.load(f)
    model.to(device)

    ####################################
    ### PREPARING EVALUATION DATASET ###
    ####################################
    sequence_length = model.sequence_length  # Ensure the model defines this
    eval_dataset = TweetDataset(X=X_eval, y=torch.zeros(len(X_eval)), sequence_length=sequence_length)
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False, drop_last=False)

    ####################
    ### EVALUATION #####
    ####################
    y_pred, _ = CNN_kaggle_eval(model, eval_loader, n_epochs=1)  # Generate predictions
    df_results = pd.DataFrame(data={
        "ID": ids,
        "EventType": y_pred
    })

    # Save predictions to CSV
    output_path = f"submissions/{path_model.split('/')[-1][:-4]}.csv"
    df_results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}", '\n')

    ##################################
    ### EVALUATION ON TRAINING SET ###
    ##################################
    
    ##########################
    ### READ TRAINING DATA ###
    ##########################
    path_dfs_train = "challenge_data/train/train_BERT_PCA"
    df_train_ = utils.read_BERT(path_dfs=path_dfs_train)
    X_train, y_train = utils.df_to_tensors(df_train_)

    ###################################
    ### EVALUATION ON TRAINING DATA ###
    ###################################
    print("Evaluating the model on the training dataset...")
    train_dataset = TweetDataset(X=X_train, y=torch.zeros(len(X_train)), sequence_length=model.sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, drop_last=False)

    y_train_pred, _ = CNN_kaggle_eval(model, train_loader, n_epochs=1)

    #########################
    ### CALCULATE METRICS ###
    #########################
    accuracy = accuracy_score(y_train, y_train_pred)
    precision = precision_score(y_train, y_train_pred, average='weighted')
    recall = recall_score(y_train, y_train_pred, average='weighted')
    f1 = f1_score(y_train, y_train_pred, average='weighted')

    #####################
    ### PRINT RESULTS ###
    #####################
    print("Classification Report on Training Data:")
    print(classification_report(y_train, y_train_pred))
    print("\nModel Performance Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}", '\n')

    ##################################
    ### NO EVALUATION ON TEST DATA ###
    ##################################
    print("No evaluation possible on test data, as we do not have the ground truth labels.", '\n')