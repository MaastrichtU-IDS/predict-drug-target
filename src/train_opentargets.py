import glob
import json
import os

import pandas as pd
from tqdm import tqdm

from src.embeddings import compute_drug_embedding, compute_target_embedding
from src.train import train, compute_and_train
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


def train_opentargets(input_dir, out_dir):
    """Main function to orchestrate the extraction and saving process."""
    known_drug_targets = []

    # first extract the drug-target pairs from the opentargets json files
    json_files = get_jsonl_files(input_dir)
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
    # print(df_known_dt)
    scores = compute_and_train(df_known_dt, out_dir)


if __name__ == "__main__":
    target_directory = "data/download/opentargets/knownDrugsAggregated"
    output_directory = "data/opentargets"  # Replace with desired output CSV file name/path
    os.makedirs(output_directory, exist_ok=True)
    train_opentargets(target_directory, output_directory)
