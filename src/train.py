"""Common functions for training the models"""
import gc
import numbers
import os
import pickle
import time
import random
import concurrent.futures
from datetime import date, datetime
from itertools import product

import numpy as np
import pandas as pd
import torch
from sklearn import ensemble, metrics
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, KFold
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, f1_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier, DMatrix

from src.embeddings import compute_drug_embedding, compute_target_embedding
from src.utils import log, TrainingConfig
from src.vectordb import init_vectordb

vectordb = init_vectordb(recreate=False)


def generate_dt_pairs(dt_df):
    """Get pairs and their labels: All given known drug-target pairs are 1,
    We add pairs for missing drug/targets combinations as 0 (not known as interacting)"""
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

    print("FIT X Y")
    print(X)
    print(y)

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
        tree_method='hist', # Use GPU optimized histogram algorithm
        # device='gpu',
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


def compute(df_known_dt: pd.DataFrame | str, out_dir: str = "data"):
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
    # return train(df_known_dt, df_drugs, df_targets, save_model=f"{out_dir}/opentarget_drug_target.pkl")
    return df_known_dt, df_drugs, df_targets


################### Train with a grid of hyperparameters to find the best


# def get_params_combinations(params):
# 	keys, values = zip(*params.items())
# 	combinations = [dict(zip(keys, v)) for v in product(*values)]
# 	return combinations

def train_gpu(
    df_known_interactions: pd.DataFrame,
    df_drugs_embeddings: pd.DataFrame,
    df_targets_embeddings: pd.DataFrame,
    params: dict[str, int | float],
    save_model: str = "models/drug_target.pkl",
):
    """Train and compare a grid of hyperparameters

    Training takes 3 dataframes as input, ideally use CURIEs for drug/target IDs:
    1. a df with known drug-target interactions (2 cols: drug, target)
    2. a df with drug embeddings: drug col + 512 cols for embeddings
    3. a df with target embeddings: target col + 1280 cols for embeddings
    """
    time_start = datetime.now()
    embeddings = {
        "drug": df_drugs_embeddings,
        "target": df_targets_embeddings,
    }

    print("Generate DT pairs")
    # Get pairs and their labels: All given known drug-target pairs are 1
    # we add pairs for missing drug/targets combinations as 0 (not known as interacting)
    pairs, labels = generate_dt_pairs(df_known_interactions)

    # TODO: Split dataset for train/test?
    # X_train, X_test, y_train, y_test = train_test_split(pairs, labels, test_size=0.2, random_state=123)

    print("Merging drug/target pairs and their labels in a DF")
    # Merge drug/target pairs and their labels in a DF
    train_df = pd.DataFrame(
        list(zip(pairs[:, 0], pairs[:, 1], labels)), columns=["drug", "target", "Class"]
    )
    print("Merging embeddings in the DF")
    # Add the embeddings to the DF
    train_df = train_df.merge(embeddings["drug"], left_on="drug", right_on="drug").merge(
        embeddings["target"], left_on="target", right_on="target"
    )

    print("Getting X and y")
    # X is the array of embeddings (drug+target), without other columns
    # y is the array of classes/labels (0 or 1)
    embedding_cols = train_df.columns.difference(["drug", "target", "Class"])
    X = train_df[embedding_cols].values
    y = train_df["Class"].values.ravel()
    print(f"Features count: {len(embedding_cols)}")
    # print(X)
    # print(y)

    ndrugs = len(embeddings["drug"])
    ntargets = len(embeddings["target"])
    _unique, counts = np.unique(labels, return_counts=True)
    ndrugtargets = counts[1]
    log.info(f"Training based on {ndrugtargets} Drug-Targets known interactions: {ndrugs} drugs | {ntargets} targets")
    random_state=123 # Or 42?
    n_jobs = 2 # Or -1
    n_splits = 5

    # pairs, classes = balance_data(pairs_all, classes_all, n_proportion)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # TODO: xgboost don't support gridsearch on GPU by default
    # https://github.com/compomics/ms2pip/blob/a8c61b41044f3f756b4551d7866d8030e68b1570/train_scripts/train_xgboost_c.py#L143

    results = []
    # combinations = get_params_combinations(params_grid)

    # NOTE: To run XGB on GPU:
    # params["device"] = "cuda:0"
    # params["tree_method"] = "hist"

    # print("Working on combination {}/{}".format(count, len(combinations)))
    combination_time = time.time()
    fold_results = []

    # for fold, (train_index, test_index) in enumerate(kf.split(X)):
    # Train model for each fold
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # # Send data to GPU for xgboost
        # send_time = time.time()
        # dtrain = xgb.DMatrix(X_train, label=y_train)
        # dtest = xgb.DMatrix(X_test, label=y_test)
        # print(f"Sending data to GPU took {time.time() - send_time}s")

        # Train xgboost model
        # model = xgb.train(params, dtrain, num_boost_round=100)

        model = RandomForestClassifier(**params)
        model.fit(x_train, y_train)

        # Evaluate model
        predictions = model.predict(x_test)
        predictions_binary = np.round(predictions) # Convert probabilities to binary outputs

        # Calculate metrics
        rmse = np.sqrt(((predictions - y_test) ** 2).mean())
        precision = precision_score(y_test, predictions_binary)
        recall = recall_score(y_test, predictions_binary)
        accuracy = accuracy_score(y_test, predictions_binary)
        roc_auc = roc_auc_score(y_test, predictions)
        f1 = f1_score(y_test, predictions_binary)
        average_precision = average_precision_score(y_test, predictions)

        fold_results.append({
            'rmse': rmse,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'f1': f1,
            'average_precision': average_precision
        })
        # rmse = np.sqrt(((predictions - y_test) ** 2).mean())
        # fold_results.append(rmse)
        # del dtrain, dtest, model
        gc.collect()  # Force garbage collection for xgb on GPU
        print(f"Completed fold {fold + 1}/{n_splits} in {time.time() - send_time}s")

    print(f"Combination took {time.time() - combination_time}s")

    # # Store the average RMSE for this parameter combination
    # avg_rmse = np.mean(fold_results)
    # results.append({'rmse': avg_rmse})
    # # count += 1
    # df = pd.DataFrame(results)

    df_avg_metrics = pd.DataFrame(fold_results).mean()
    print("TRAINING RESULTS")
    print(df_avg_metrics)
    return df_avg_metrics

    #######################


    # NOTE: xgb.cv() does not work on GPU, but xgb.train() works on GPU
    # for param_combin in combinations:
    #     # param_combin["n_jobs"] = n_jobs
    #     # param_combin["random_state"] = random_state
    #     # param_combin["tree_method"] = "gpu_hist"
    #     param_combin["device"] = "cuda:0"
    #     param_combin["tree_method"] = "hist"

    #     print("Working on combination {}/{}".format(count, len(combinations)))
    #     start_time = time.time()
    #     count += 1

    #     # Run cross validation on GPU
    #     tmp = xgb.train(param_combin, dtrain, num_boost_round=100)
    #     # tmp = xgb.cv(param_combin, dtrain, nfold=5)
    #     # tmp = xgb.cv(param_combin, dtrain, nfold=5, num_boost_round=100, early_stopping_rounds=10, verbose_eval=10)

    #     print(f"Took {time.time() - start_time}s")
    #     tmp['boosting-round'] = tmp.index
    #     for param in param_combin.keys():
    #         tmp[param] = param_combin[param]
    #     result = result.append(tmp)
    # print("result!", result)


    #######################

    # # Create a KFold object for cross-validation
    # kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    # grid_search = GridSearchCV(estimator=xgb_model, param_grid=params_grid, scoring='f1', cv=kf, n_jobs=n_jobs)
    # # or scoring='accuracy'

    # log.info("Fitting grid search")
    # grid_search.fit(X, y)
    # grid_search

    # # Without CV:temp_folder
    # # grid_search = GridSearchCV(estimator=xgb_model, param_grid=params_grid, scoring='accuracy', cv=5, n_jobs=n_jobs)

    # # Perform grid search on the training data
    # # grid_search.fit(X_train, y_train)

    # # Print the best parameters and the corresponding accuracy
    # log.info("Best Parameters:", grid_search.best_params_)
    # log.info("Best Accuracy:", grid_search.best_score_)

    # # Creating DataFrame from cv_results
    # results_df = pd.DataFrame(grid_search.cv_results_)
    # results_df = results_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]

    # # Evaluate on test data
    # best_model = grid_search.best_estimator_

    # test_accuracy = best_model.score(X_test, y_test)
    # log.info("Test Accuracy:", test_accuracy)

    # log.info(f"⚡ Training took {datetime.now() - time_start}")

    # os.makedirs("models", exist_ok=True)
    # with open(save_model, "wb") as f:
    #     pickle.dump(best_model, f) #rf_model

    # return results_df
    # # return agg_df.to_dict(orient="records")

