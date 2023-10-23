import csv
import glob
import json
import os

import pandas as pd
import requests
from tqdm import tqdm
import logging

from src.embeddings import compute_drug_embedding, compute_target_embedding
from src.utils import COLLECTIONS, log, get_smiles_for_drug, get_seq_for_target, get_pref_ids
from src.vectordb import init_vectordb

# NOTE: Download opentargets before running this script
# ./scripts/download_opentargets.sh

# We need to generate 3 dataframe/CSV to provide to training as input:
# 1. a known drug-target interactions df (2 cols)
# 2. a df with drug embeddings: drug col + 512 cols for embeddings
# 2. a df with target embeddings: target col + 1280 cols for embeddings


# A list of KNOWN drugs-interacts_with-targets (from opentarget)
# Once we have this list, we just need to pass it to the compute_drug_embedding or compute_target_embedding functions
# These functions returns a dataframe with a "drug" column for the ID, and all other columns are the embeddings
# knownInteraction [ drug_id - target_id - 0 or 1 if interacts or not] (or could even be the mechanism of action string)
# target_df[id - embeddings]
# drugs_df[id - embeddings]
# TODO: First we get the df of knownInteraction, then generate list of drugs, pass it to function to calculate embed, same for targets


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

    vectordb = init_vectordb(COLLECTIONS, recreate=False)

    df_known_dt = pd.DataFrame(known_drug_targets)


    # These functions retrieves SMILES and compute embeddings in 1 batch
    df_drugs = compute_drug_embedding(vectordb, set(df_known_dt["drug"].tolist()), tmp_dir="data/opentargets/")
    log.info("DRUGS EMBEDDINGS COMPUTED")
    df_targets = compute_target_embedding(vectordb, set(df_known_dt["target"].tolist()), tmp_dir="data/opentargets/")
    log.info("TARGETS EMBEDDINGS COMPUTED")

    df_drugs.to_csv('data/opentargets/drugs_embeddings.csv', index=False)
    df_targets.to_csv('data/opentargets/targets_embeddings.csv', index=False)

    # TODO: remove from df_known_dt entries where we don't have SMILES or AA seq



if __name__ == "__main__":
    target_directory = "data/download/opentargets/knownDrugsAggregated"
    output_directory = "data/processed"  # Replace with desired output CSV file name/path
    os.makedirs(output_directory, exist_ok=True)
    prepare(target_directory, output_directory)
