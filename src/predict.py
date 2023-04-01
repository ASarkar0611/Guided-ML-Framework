import os

import config
import dispatcher

import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics

import argparse
import joblib

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
MODEL = dispatcher.MODELS

def predict(model,fold):
    df = pd.read_csv(config.TEST_DATA)
    test_idx = df["id"].values
    predictions = None

    for FOLD in range(5):
        df = pd.read_csv(config.TEST_DATA)
        print(df.head(3))
        encoders = joblib.load(os.path.join("../models/", f"{args.model}_{fold}_label_encoders.pkl"))
        cols = joblib.load(os.path.join("../models/", f"{args.model}_{fold}_columns.pkl"))
        #print(cols)
        for c in encoders:
            print(c)
            lbl = encoders[c]
            df.loc[:, c] = df.loc[:, c].astype(str).fillna("NONE")
            df.loc[:, c] = lbl.transform(df[c].values.tolist())
            
            clf = joblib.load(os.path.join("../models/", f"{args.model}_{fold}.pkl"))
            print("cols =", cols)
            df = df[cols]
            
            preds = clf.predict_proba(df)[:, 1]
            
            if fold == 0:
                predictions = preds
            else:
                predictions += preds
    
    predictions /= 5
    #print(predictions)
    sub = pd.DataFrame(np.column_stack((test_idx, predictions), columns=["id", "target"]))
    return sub

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

    submission = predict(args.model, args.fold)
    submission.loc[:, "id"] = submission.loc[:, "id"].astype(int)
    submission.to_csv(f"../models/{args.model}.csv", index=False)