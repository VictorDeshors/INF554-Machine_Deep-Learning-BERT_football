#%% Imports
""" Script to add new features and do a PCA on the BERT embeddings """

# Add the parent directory to the sys.path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pandas as pd
import src.utils as utils
from sklearn.decomposition import PCA
import re
from sklearn.preprocessing import MinMaxScaler


def add_tweet_features(df):
    """Add new features to the dataframe df"""

    regex_repeated_letters = r"(.)\1{2,}"
    df["Has_Repeated_Letters"] = df["Tweet"].str.contains(regex_repeated_letters)

    emoji_regex = r"[\U00010000-\U0010ffff]"
    df["Emoji_Count"] = df["Tweet"].apply(lambda x: len(re.findall(emoji_regex, x)))

    df["Exclamation_Count"] = df["Tweet"].apply(lambda x: x.count("!"))
    df["Question_Count"] = df["Tweet"].apply(lambda x: x.count("?"))

    df["starts_with_RT"] = df["Tweet"].str.startswith("RT")
    df["isMention"] = df["Tweet"].str.contains("@")

    agg_features = {
        "Has_Repeated_Letters": "sum",
        "Emoji_Count": "sum",
        "Exclamation_Count": "sum",
        "Question_Count": "sum",
        "starts_with_RT": "sum",
        "isMention": "sum",
    }

    df_agg = df.groupby("ID").agg(agg_features).reset_index()

    # Calculate features per minute
    df_agg["nb_tweets_per_minute"] = df.groupby("ID")["ID"].transform("count")
    df_agg["nb_consecutive_letters_per_minute"] = df_agg["Has_Repeated_Letters"]
    df_agg["nb_smileys_per_minute"] = df_agg["Emoji_Count"]
    df_agg["Exclamation_Count_per_minute"] = df_agg["Exclamation_Count"]
    df_agg["Question_Count_per_minute"] = df_agg["Question_Count"]
    df_agg["nb_RT_per_min"] = df_agg["starts_with_RT"]
    df_agg["nb_@_per_min"] = df_agg["isMention"]
    df_agg["Match_time"] = df_agg["ID"].str.split("_").str[1].astype(int)

    # Define features to normalize
    features_to_normalize = [
        "nb_tweets_per_minute",
        "nb_consecutive_letters_per_minute",
        "nb_smileys_per_minute",
        "Exclamation_Count_per_minute",
        "Question_Count_per_minute",
        "nb_RT_per_min",
        "nb_@_per_min",
    ]

    scaler = MinMaxScaler()
    df_normalized = df_agg.copy()
    df_normalized[features_to_normalize] = scaler.fit_transform(
        df_normalized[features_to_normalize]
    )
    df_new_features = df_normalized.drop_duplicates(subset="ID")

    return df_new_features





if __name__ == "__main__":

    ########################
    ### ARGUMENT PARSING ###
    ########################

    if len(sys.argv) <= 1:
        print("Please provide an argument: 'eval' or 'train'")
        sys.exit(1)
    
    mode = sys.argv[1]

    if mode == "train":
        path_tweets = "challenge_data/train/train_tweets"
        path_embeddings = "challenge_data/train/train_BERT"
        path_save = "challenge_data/train/train_BERT_PCA/df_BERT_PCA.pkl"

    else:
        path_tweets = "challenge_data/eval/eval_tweets"
        path_embeddings = "challenge_data/eval/eval_BERT"
        path_save = "challenge_data/eval/eval_BERT_PCA/df_BERT_PCA.pkl"


    #######################
    ### READ EMBEDDINGS ###
    #######################
    df_BERT = utils.read_BERT(path_dfs=path_embeddings).drop(columns="Timestamp")


    ####################
    ### NEW FEATURES ###
    ####################

    files = []

    for i, filename in enumerate(os.listdir(path_tweets)):
        if filename == ".ipynb_checkpoints":
            continue
        df = pd.read_csv(f"{path_tweets}/" + filename)
        files.append(df)

    df = pd.concat(files, ignore_index=True)
    df_new_features = add_tweet_features(df)

    ##########################################################
    ### JOIN TRAINING SETs WITH EMBEDDINGS AND NEW FEATURES ###
    ##########################################################

    df_BERT["ID"] = df_BERT["ID"].astype(str)
    df_new_features["ID"] = df_new_features["ID"].astype(str)

    columns_to_keep = [
        "nb_tweets_per_minute",
        "nb_RT_per_min",
        "nb_@_per_min",
        "Match_time",
        "nb_consecutive_letters_per_minute",
        "nb_smileys_per_minute",
        "Exclamation_Count_per_minute",
        "Question_Count_per_minute",
    ]

    df_BERT_ =  df_BERT.set_index("ID").join(df_new_features.set_index("ID")[columns_to_keep]).reset_index()


    ###########
    ### PCA ###
    ###########


    # Columns to apply PCA on
    columns_to_pca = [i for i in range(0, 768)]  # Columns '0' to '767'
    X_pca_input = df_BERT_[columns_to_pca]

    # Number of Principal Components
    N = 50

    pca = PCA(n_components=N)  
    X_pca = pca.fit_transform(X_pca_input)
    pca_columns = [f"PCA_{i+1}" for i in range(N)]
    df_pca = pd.DataFrame(X_pca, columns=pca_columns, index=df_BERT_.index)

    columns_to_keep = [col for col in df_BERT_.columns if col not in columns_to_pca]
    df_BERT_ = pd.concat([df_BERT_[columns_to_keep], df_pca], axis=1)


    #################
    ### Save data ###
    #################

    df_BERT_.to_pickle(path_save)
    print(f"Data saved in {path_save}")