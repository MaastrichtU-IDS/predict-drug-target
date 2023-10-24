import glob
import json
import os

import pandas as pd
from tqdm import tqdm

from src.embeddings import compute_drug_embedding, compute_target_embedding
from src.train import train
from src.utils import COLLECTIONS, log
from src.vectordb import init_vectordb

# NOTE: Download opentargets before running this script
# ./scripts/download_opentargets.sh

# Output file path
output_file_path = "../data/opentargets/merged_parsed.csv"


def get_jsonl_files(target_directory):
    """Return a list of JSONL files from the target directory."""
    return glob.glob(os.path.join(target_directory, "*.json"))


def extract_data_from_jsonl(filename):
    """Extract drugId and targetId from a JSONL file."""
    with open(filename) as file:
        for line in file:
            data = json.loads(line.strip())
            yield data.get("drugId", None), data.get("targetId", None)


def prepare(target_directory, output_directory):
    """Main function to orchestrate the extraction and saving process."""
    known_drug_targets = []

    # first extract the drug-target pairs from the opentargets json files
    json_files = get_jsonl_files(target_directory)
    for json_file in tqdm(json_files, desc="Processing files"):
        # log.info(json_file)
        for drugId, targetId in extract_data_from_jsonl(json_file):
            known_drug_targets.append(
                {
                    "drug": f"CHEMBL.COMPOUND:{drugId}",
                    "target": f"ENSEMBL:{targetId}",
                }
            )
    df_known_dt = pd.DataFrame(known_drug_targets)

    vectordb = init_vectordb(COLLECTIONS, recreate=False)

    # These functions retrieves SMILES and compute embeddings in 1 batch
    df_drugs = compute_drug_embedding(vectordb, set(df_known_dt["drug"].tolist()), tmp_dir="data/opentargets/")
    df_drugs.to_csv("data/opentargets/drugs_embeddings.csv", index=False)
    log.info("DRUGS EMBEDDINGS COMPUTED")

    df_targets = compute_target_embedding(vectordb, set(df_known_dt["target"].tolist()), tmp_dir="data/opentargets/")
    df_targets.to_csv("data/opentargets/targets_embeddings.csv", index=False)
    log.info("TARGETS EMBEDDINGS COMPUTED")

    # Remove from df_known_dt entries where we don't have SMILES or AA seq
    df_known_dt = df_known_dt.merge(df_drugs[["drug"]], on="drug").merge(df_drugs[["target"]], on="target")

    # Run the training
    log.info("Start training")
    train(df_known_dt, df_drugs, df_targets, "models/opentargets_drug_target.pkl")


if __name__ == "__main__":
    target_directory = "data/download/opentargets/knownDrugsAggregated"
    output_directory = "data/processed"  # Replace with desired output CSV file name/path
    os.makedirs(output_directory, exist_ok=True)
    prepare(target_directory, output_directory)
