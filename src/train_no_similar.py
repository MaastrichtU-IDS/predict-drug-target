import os
import pandas as pd

from src.train import train
from src.utils import log, TrainingConfig
from src.vectordb import init_vectordb


# NOTE: script to test various config while training the model
# For speed, it DOES NOT compute embeddings for all drugs and targets
# It expects the embeddings to be already generated in a CSV (by train_opentargets.py)


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



def train_many_configs(input_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    subject_sim_thresholds = [1, 0.99, 0.98, 0.97, 0.95, 0.90]
    object_sim_thresholds = [1, 0.99, 0.98, 0.97, 0.95, 0.90]
    cv_nfolds = [10, 20]
    max_depths = [6, -1]

    configs = []
    for subject_sim_threshold in subject_sim_thresholds:
        for object_sim_threshold in object_sim_thresholds:
            for cv_nfold in cv_nfolds:
                for max_depth in max_depths:
                    configs.append(TrainingConfig(
                        subject_sim_threshold=subject_sim_threshold,
                        object_sim_threshold=object_sim_threshold,
                        cv_nfold=cv_nfold,
                        max_depth=max_depth,
                    ))

    score_list = []
    for config in configs:
        score_list.append(train_not_similar(input_dir, out_dir, config))

    print(score_list)

    combined_df = pd.concat(score_list)

    combined_df.to_csv(f"{out_dir}/compare_scores.csv", index=False)


def train_not_similar(input_dir, out_dir, config: TrainingConfig):
    os.makedirs(out_dir, exist_ok=True)

    print(f"🔨 Training for {config}")

    df_known_dt = pd.read_csv(f"{input_dir}/known_drugs_targets.csv")

    df_drugs = pd.read_csv(f"{input_dir}/drugs_embeddings.csv")
    df_targets = pd.read_csv(f"{input_dir}/targets_embeddings.csv")

    log.info(f"DF LENGTH BEFORE DROPPING: {len(df_drugs)} drugs and {len(df_targets)} targets, and {len(df_known_dt)} known pairs")

    df_drugs = drop_similar(df_drugs, "drug", config.subject_sim_threshold)
    df_targets = drop_similar(df_targets, "target", config.object_sim_threshold)

    df_known_dt = df_known_dt.merge(df_drugs[["drug"]], on="drug").merge(df_targets[["target"]], on="target")

    log.info(f"DF LENGTH AFTER DROPPING: {len(df_drugs)} drugs and {len(df_targets)} targets, and {len(df_known_dt)} known pairs")

    score_df = train(df_known_dt, df_drugs, df_targets, save_model=f"{out_dir}/opentarget_drug_target_nosim.pkl", config=config)
    score_df.insert(0, 'Drug sim threshold', config.subject_sim_threshold)
    score_df.insert(1, 'Target sim threshold', config.object_sim_threshold)
    score_df.insert(2, 'CV nfold', config.cv_nfold)
    score_df.insert(3, 'Max depth', config.max_depth)

    return score_df


if __name__ == "__main__":
    train_many_configs("data/opentargets", "data/results")
    # train_not_similar("data/opentargets", "data/opentargets_not_similar")
