import os

import config
import dispatcher
import argparse

import pandas as pd
#from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics

import joblib

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
#FOLD = int(os.environ.get("FOLD"))
FOLD = config.FOLD
MODEL = dispatcher.MODELS
#print(MODEL)

FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}
#print(FOLD_MAPPING.get(FOLD))

def run(model,fold):
    df = pd.read_csv(config.TRAINING_DATA)
    df_test = pd.read_csv(config.TEST_DATA)
    #print(FOLD_MAPPING[FOLD])
    #train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    #valid_df = df[df.kfold==FOLD]

    train_df = df[df.kfold==fold]
    valid_df = df[df.kfold==fold]

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df.drop(["id", "target", "kfold"], axis=1)
    valid_df = valid_df.drop(["id", "target", "kfold"], axis=1)

    valid_df = valid_df[train_df.columns] # to keep the order of the columns same in train and valid df

    label_encoders = {}
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        # train_df.loc[:, c] = train_df.loc[:, c].astype(str).fillna("NONE")
        # valid_df.loc[:, c] = valid_df.loc[:, c].astype(str).fillna("NONE")
        # df_test.loc[:, c] = df_test.loc[:, c].astype(str).fillna("NONE")
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist() + df_test[c].values.tolist())
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders[c] = lbl
    
    # data is ready to train
    #print(MODEL[args.model])
    clf = MODEL[args.model]
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:, 1]
    #print(preds)

    print(metrics.roc_auc_score(yvalid, preds)) # when data set is skewed we are using auc score

    joblib.dump(label_encoders, f"../models/{args.model}_{fold}_label_encoders.pkl")
    joblib.dump(clf, f"../models/{args.model}_{fold}.pkl")
    joblib.dump(train_df.columns, f"../models/{args.model}_{fold}_columns.pkl")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int
    )

    parser.add_argument(
        "--model",
        type=str
    )

    args = parser.parse_args()

    run(model=args.model,
        fold=args.fold)
