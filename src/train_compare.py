import os
import pandas as pd

from src.train import train, train_grid
from src.utils import log, TrainingConfig
from src.vectordb import init_vectordb


# NOTE: script to test various config while training the model
# For speed, it DOES NOT compute embeddings for all drugs and targets
# It expects the embeddings to be already generated in a CSV (by train_compute.py)


def drop_similar(df: str, col_id: str, threshold: float = 0.9):
    """Given a DF remove all entities that are too similar"""
    vectordb = init_vectordb(recreate=False)
    indices_to_drop = []
    # TODO: remove things that are too similar
    # in df_drugs and df_targets
    for i, row in df.iterrows():
        if row[col_id] in indices_to_drop:
            # If we already plan to drop this row, skip it
            continue
        # The column ID and the collection are the same (drug or target)
        ent_matching = vectordb.get(col_id, row[col_id])
        if ent_matching:
            # Find vectors that are similar to the vector of the given drug ID
            search_res = vectordb.search(col_id, ent_matching[0].vector)

            for res in search_res:
                if threshold < res.score < 1:
                    indices_to_drop.append(res.payload['id'])
                    # print(f"PLEASE REMOVE {res.score}")
                    df = df[df[col_id] != res.payload['id']]
                # else:
                #     print(f"DONT REMOVE {res.score}")
                # print(f"{res.payload['id']}: {res.score} ({res.id})")
        else:
            print(f"No match for {row[col_id]}")
    log.info(f"DROPPING {col_id}: {len(indices_to_drop)}")
    # return df.drop(indices_to_drop)
    return df



def exclude_sim_and_train(input_dir, out_dir, param_grid, subject_sim_threshold: float = 1, object_sim_threshold: float = 1):
    """Exclude similarities given thresholds, and run training on grid"""
    os.makedirs(out_dir, exist_ok=True)

    print(f"🔨 Training for {subject_sim_threshold} - {object_sim_threshold}")

    # Precomputed embeddings
    df_known_dt = pd.read_csv(f"{input_dir}/known_drugs_targets.csv")
    df_drugs = pd.read_csv(f"{input_dir}/drugs_embeddings.csv")
    df_targets = pd.read_csv(f"{input_dir}/targets_embeddings.csv")

    log.info(f"DF LENGTH BEFORE DROPPING: {len(df_drugs)} drugs and {len(df_targets)} targets, and {len(df_known_dt)} known pairs")

    if subject_sim_threshold < 1:
        df_drugs = drop_similar(df_drugs, "drug", subject_sim_threshold)

    if object_sim_threshold < 1:
        df_targets = drop_similar(df_targets, "target", object_sim_threshold)

    df_known_dt = df_known_dt.merge(df_drugs[["drug"]], on="drug").merge(df_targets[["target"]], on="target")

    log.info(f"DF LENGTH AFTER DROPPING: {len(df_drugs)} drugs and {len(df_targets)} targets, and {len(df_known_dt)} known pairs")

    score = train_grid(df_known_dt, df_drugs, df_targets, param_grid, f"{out_dir}/model_drug_target_{subject_sim_threshold}_{object_sim_threshold}.pkl")

    # score_df = train(df_known_dt, df_drugs, df_targets, save_model=f"{out_dir}/opentarget_drug_target_nosim.pkl", config=config)
    # score_df.insert(0, 'Drug sim threshold', config.subject_sim_threshold)
    # score_df.insert(1, 'Target sim threshold', config.object_sim_threshold)
    # score_df.insert(2, 'CV nfold', config.cv_nfold)
    # score_df.insert(3, 'Max depth', config.max_depth)

    return score


def train_grid_exclude_sim(input_dir, out_dir):
    """Define the similarities thresholds and params grid, then run training"""
    os.makedirs(out_dir, exist_ok=True)
    # Shorter version for starting
    subject_sim_thresholds = [1, 0.99]
    object_sim_thresholds = [1, 0.99]
    param_grid = {
        'max_depth': [3, 4],
        'learning_rate': [0.1, 0.01],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8],
        'gamma': [0, 1],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 2],
        # 'n_estimators': [100, 200],
    }

    # Longer version
    # subject_sim_thresholds = [1, 0.99, 0.98, 0.97, 0.95, 0.90]
    # object_sim_thresholds = [1, 0.99, 0.98, 0.97, 0.95, 0.90]
    # param_grid = {
    #     'max_depth': [3, 4, 5, 6],
    #     'learning_rate': [0.1, 0.01, 0.05],
    #     'subsample': [0.7, 0.8, 0.9],
    #     'colsample_bytree': [0.7, 0.8, 0.9],
    #     'gamma': [0, 1, 2],
    #     'reg_alpha': [0, 0.1, 0.5],
    #     'reg_lambda': [1, 2, 5],
    #     'n_estimators': [100, 200, 300],
    # }

    scores_df = pd.DataFrame()
    for subject_sim_threshold in subject_sim_thresholds:
        for object_sim_threshold in object_sim_thresholds:
            sim_scores = exclude_sim_and_train(input_dir, out_dir, param_grid, subject_sim_threshold, object_sim_threshold)
            sim_scores["subject_sim_threshold"] = subject_sim_threshold
            sim_scores["object_sim_threshold"] = object_sim_threshold
            scores_df = pd.concat([scores_df, sim_scores], ignore_index=True)

    # score_list = []
    # for config in configs:
    #     score_list.append(train_not_similar(input_dir, out_dir, config))
    # print(score_list)
    # combined_df = pd.concat(score_list)
    # combined_df.to_csv(f"{out_dir}/compare_scores.csv", index=False)

    print("SCORES DF", scores_df)
    scores_df.to_csv(f"{out_dir}/compare_scores.csv", index=False)



if __name__ == "__main__":
    train_grid_exclude_sim("data/opentargets", "data/grid")
    # train_not_similar("data/opentargets", "data/opentargets_not_similar")
