import glob
import json
import os

import pandas as pd

from src.train import train
from src.utils import COLLECTIONS, log
from src.vectordb import init_vectordb


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




def train_not_similar(input_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    df_known_dt = pd.read_csv(f"{input_dir}/known_drugs_targets.csv")

    df_drugs = pd.read_csv(f"{input_dir}/drugs_embeddings.csv")
    df_targets = pd.read_csv(f"{input_dir}/targets_embeddings.csv")

    log.info(f"DF LENGTH BEFORE DROPPING: {len(df_drugs)} drugs and {len(df_targets)} targets, and {len(df_known_dt)} known pairs")

    df_drugs = drop_similar(df_drugs, "drug", 0.9)
    df_targets = drop_similar(df_targets, "target", 0.98)

    df_known_dt = df_known_dt.merge(df_drugs[["drug"]], on="drug").merge(df_targets[["target"]], on="target")

    log.info(f"DF LENGTH AFTER DROPPING: {len(df_drugs)} drugs and {len(df_targets)} targets, and {len(df_known_dt)} known pairs")

    return train(df_known_dt, df_drugs, df_targets, save_model=f"{out_dir}/opentarget_drug_target_nosim.pkl")


if __name__ == "__main__":
    train_not_similar("data/opentargets", "data/opentargets_not_similar")
