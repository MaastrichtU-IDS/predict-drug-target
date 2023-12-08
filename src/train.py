import numbers
import os
import pickle
import random
import concurrent.futures
from datetime import date

import numpy as np
import pandas as pd
import torch
from sklearn import ensemble, metrics
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.embeddings import compute_drug_embedding, compute_target_embedding
from src.utils import log, TrainingConfig
from src.vectordb import init_vectordb

vectordb = init_vectordb(recreate=False)


def generate_dt_pairs(dt_df):
    dtKnown = {tuple(x) for x in dt_df[["drug", "target"]].values}
    pairs = []
    labels = []

    drugs = set(dt_df.drug.unique())
    targets = set(dt_df.target.unique())
    for d in drugs:
        for t in targets:
            label = 1 if (d, t) in dtKnown else 0

            pairs.append((d, t))
            labels.append(label)

    pairs = np.array(pairs)
    labels = np.array(labels)
    return pairs, labels


def multimetric_score(estimator, X_test, y_test, scorers):
    """Return a dict of score for multimetric scoring"""
    scores = {}
    for name, scorer in scorers.items():
        score = scorer(estimator, X_test) if y_test is None else scorer(estimator, X_test, y_test)

        if hasattr(score, "item"):
            try:
                # e.g. unwrap memmapped scalars
                score = score.item()
            except ValueError:
                # non-scalar?
                pass
        scores[name] = score

        if not isinstance(score, numbers.Number):
            raise ValueError(
                f"scoring must return a number, got {score!s} ({type(score)}) " f"instead. (scorer={name})"
            )
    return scores


def balance_data(pairs, classes, n_proportion):
    classes = np.array(classes)
    pairs = np.array(pairs)

    indices_true = np.where(classes == 1)[0]
    indices_false = np.where(classes == 0)[0]

    np.random.shuffle(indices_false)
    indices = indices_false[: (n_proportion * indices_true.shape[0])]

    print(f"True positives: {len(indices_true)}")
    print(f"True negatives: {len(indices_false)}")
    pairs = np.concatenate((pairs[indices_true], pairs[indices]), axis=0)
    classes = np.concatenate((classes[indices_true], classes[indices]), axis=0)

    return pairs, classes


def get_scores(clf, X_new, y_new):
    scoring = ["precision", "recall", "accuracy", "roc_auc", "f1", "average_precision"]
    scorers = metrics._scorer._check_multimetric_scoring(clf, scoring=scoring)
    scores = multimetric_score(clf, X_new, y_new, scorers)
    return scores


def crossvalid(train_df, test_df, clfs, run_index, fold_index):
    features_cols = train_df.columns.difference(["drug", "target", "Class"])
    print(f"Features count: {len(features_cols)}")
    X = train_df[features_cols].values
    y = train_df["Class"].values.ravel()

    X_new = test_df[features_cols].values
    y_new = test_df["Class"].values.ravel()

    results = pd.DataFrame()
    for name, clf in clfs:
        clf.fit(X, y)
        row = {}
        row["run"] = run_index
        row["fold"] = fold_index
        row["method"] = name
        scores = get_scores(clf, X_new, y_new)
        row.update(scores)

        df = pd.DataFrame.from_dict([row])
        results = pd.concat([results, df], ignore_index=True)

    return results  # , sclf_scores


def cv_run(run_index, pairs, classes, embedding_df, train, test, fold_index, clfs):
    # print( f"Run: {run_index} Fold: {fold_index} Train size: {len(train)} Test size: {len(test)}")
    train_df = pd.DataFrame(
        list(zip(pairs[train, 0], pairs[train, 1], classes[train])), columns=["drug", "target", "Class"]
    )
    test_df = pd.DataFrame(
        list(zip(pairs[test, 0], pairs[test, 1], classes[test])), columns=["drug", "target", "Class"]
    )

    train_df = train_df.merge(embedding_df["drug"], left_on="drug", right_on="drug").merge(
        embedding_df["target"], left_on="target", right_on="target"
    )
    test_df = test_df.merge(embedding_df["drug"], left_on="drug", right_on="drug").merge(
        embedding_df["target"], left_on="target", right_on="target"
    )

    all_scores = crossvalid(train_df, test_df, clfs, run_index, fold_index)
    if run_index == 1 and fold_index == 0:
        print(", ".join(all_scores.columns))
    print(all_scores.to_string(header=False, index=False))

    return all_scores


def cv_distribute(run_index, pairs, classes, cv, embedding_df, clfs):
    all_scores = pd.DataFrame()
    for fold in cv:
        scores = cv_run(run_index, pairs, classes, embedding_df, fold[0], fold[1], fold[2], clfs)
        all_scores = pd.concat([all_scores, scores], ignore_index=True)

    return all_scores


def kfold_cv(pairs_all, classes_all, embedding_df, clfs, n_run, n_fold, n_proportion, n_seed):
    scores_df = pd.DataFrame()
    for r in range(1, n_run + 1):
        n_seed += r
        random.seed(n_seed)
        np.random.seed(n_seed)
        pairs, classes = balance_data(pairs_all, classes_all, n_proportion)

        skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=n_seed)
        cv = skf.split(pairs, classes)

        pairs_classes = (pairs, classes)
        cv_list = [(train, test, k) for k, (train, test) in enumerate(cv)]

        scores = cv_distribute(r, pairs_classes[0], pairs_classes[1], cv_list, embedding_df, clfs)
        scores_df = pd.concat([scores_df, scores], ignore_index=True)
    return scores_df


###### Main training function


def train(
    df_known_interactions: pd.DataFrame,
    df_drugs_embeddings: pd.DataFrame,
    df_targets_embeddings: pd.DataFrame,
    save_model: str = "models/drug_target.pkl",
    config: TrainingConfig | None = None
):
    """Training takes 3 dataframes as input, ideally use CURIEs for drug/target IDs:
    1. a df with known drug-target interactions (2 cols: drug, target)
    2. a df with drug embeddings: drug col + 512 cols for embeddings
    3. a df with target embeddings: target col + 1280 cols for embeddings
    """
    if not config:
        config = TrainingConfig()

    embeddings = {
        "drug": df_drugs_embeddings,
        "target": df_targets_embeddings,
    }

    today = date.today()
    results_file = f"./data/results/drugbank_drug_targets_scores_{today}.csv"
    agg_results_file = f"./data/results/drugbank_drug_targets_agg_{today}.csv"

    # Get pairs
    pairs, labels = generate_dt_pairs(df_known_interactions)
    ndrugs = len(embeddings["drug"])
    ntargets = len(embeddings["target"])
    unique, counts = np.unique(labels, return_counts=True)
    ndrugtargets = counts[1]
    log.info(f"Training based on {ndrugtargets} Drug-Targets known interactions: {ndrugs} drugs | {ntargets} targets")

    # nb_model = GaussianNB()
    # lr_model = linear_model.LogisticRegression()
    # rf_model = ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1)
    # rf_model = ensemble.RandomForestClassifier(
    #     n_estimators=200,
    #     criterion="log_loss",
    #     max_depth=config.max_depth,
    #     min_samples_split=2,
    #     min_samples_leaf=1,
    #     max_features="sqrt",
    #     n_jobs=-1,
    # )
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=config.max_depth,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        objective='binary:logistic',  # For binary classification
        n_jobs=-1,
        random_state=42,
        tree_method='gpu_hist', # Use GPU optimized histogram algorithm
    )

    # clfs = [('Naive Bayes',nb_model),('Logistic Regression',lr_model),('Random Forest',rf_model)]
    clfs = [("XGBoost", xgb_model)] # "Random Forest", rf_model

    n_seed = 100
    n_fold = config.cv_nfold
    n_run = 2
    n_proportion = 1

    # Run training
    all_scores_df = kfold_cv(pairs, labels, embeddings, clfs, n_run, n_fold, n_proportion, n_seed)
    all_scores_df.to_csv(results_file, sep=",", index=False)

    agg_df = all_scores_df.groupby(["method", "run"]).mean().groupby("method").mean()
    agg_df.to_csv(agg_results_file, sep=",", index=False)
    log.info("Aggregated results:")
    print(agg_df)

    os.makedirs("models", exist_ok=True)
    with open(save_model, "wb") as f:
        pickle.dump(xgb_model, f) #rf_model

    return agg_df
    # return agg_df.to_dict(orient="records")


def compute_and_train(df_known_dt: pd.DataFrame | str, out_dir: str = "data"):
    """Compute embeddings and train model to predict interactions for a dataframe with 2 cols: drug, target"""
    if isinstance(df_known_dt, str):
        df_known_dt = pd.read_csv(df_known_dt)

    # These functions retrieves SMILES and compute embeddings in 1 batch
    log.info("Running drug and target embeddings computing in parallel")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit the drug and target embeddings calculation to the executor
        future_drugs = executor.submit(compute_drug_embedding, vectordb, set(df_known_dt["drug"].tolist()), out_dir)
        future_targets = executor.submit(compute_target_embedding, vectordb, set(df_known_dt["target"].tolist()), out_dir)
        # Get the results
        df_drugs = future_drugs.result()
        df_targets = future_targets.result()

    # Save result to CSV
    # df_drugs = compute_drug_embedding(vectordb, set(df_known_dt["drug"].tolist()), tmp_dir=out_dir)
    df_drugs.to_csv(f"{out_dir}/drugs_embeddings.csv", index=False)
    log.info(f"Drugs embeddings saved to {out_dir}")

    # df_targets = compute_target_embedding(vectordb, set(df_known_dt["target"].tolist()), tmp_dir=out_dir)
    df_targets.to_csv(f"{out_dir}/targets_embeddings.csv", index=False)
    log.info("Targets embeddings saved to {out_dir}")

    # Remove from df_known_dt entries where we don't have SMILES or AA seq
    known_dt_before = len(df_known_dt)
    df_known_dt = df_known_dt.merge(df_drugs[["drug"]], on="drug").merge(df_targets[["target"]], on="target")
    log.info(
        f"Number of known interactions before and after removing rows for which we don't have smiles/sequence: {known_dt_before} > {len(df_known_dt)}"
    )
    df_known_dt.to_csv(f"{out_dir}/known_drugs_targets.csv", index=False)

    # Run the training
    log.info("Start training")
    return train(df_known_dt, df_drugs, df_targets, save_model=f"{out_dir}/opentarget_drug_target.pkl")
