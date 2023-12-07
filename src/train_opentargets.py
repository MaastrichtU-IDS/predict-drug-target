import glob
import json
import os

import pandas as pd
from tqdm import tqdm

from src.embeddings import compute_drug_embedding, compute_target_embedding
from src.train import train, compute_and_train
from src.utils import COLLECTIONS, log, get_pref_ids
from src.vectordb import init_vectordb

# NOTE: script to run the WHOLE pipeline on opentargets data
# it will automatically compute embeddings for all drugs and targets
# Download opentargets before running this script: ./scripts/download_opentargets.sh

# Output file path
output_file_path = "../data/opentargets/merged_parsed.csv"


def get_jsonl_files(target_directory) -> list[str]:
    """Return a list of JSONL files from the target directory."""
    return glob.glob(os.path.join(target_directory, "*.json"))


def extract_data_from_jsonl(filename):
    """Extract drugId and targetId from a JSONL file."""
    with open(filename) as file:
        for line in file:
            data = json.loads(line.strip())
            yield data.get("drugId", None), data.get("targetId", None)



def ensembl_to_uniprot():
    """Dict to convert ENSEMBL IDs to UniProt IDs"""
    json_files = get_jsonl_files("data/download/opentargets/targets")
    ensembl_to_uniprot_dict = {}

    for json_file in tqdm(json_files, desc="Mapping targets ENSEMBL IDs to UniProt"):
        with open(json_file) as file:
            for line in file:
                data = json.loads(line.strip())
                for prot in data.get("proteinIds", []):
                    if prot["source"] == "uniprot_swissprot":
                        ensembl_to_uniprot_dict[data["id"]] = f"UniProtKB:{prot['id']}"

    return ensembl_to_uniprot_dict



def train_opentargets(input_dir, out_dir):
    """Main function to orchestrate the extraction and saving process."""
    os.makedirs(out_dir, exist_ok=True)
    known_drug_targets = []

    ensembl_to_uniprot_dict = ensembl_to_uniprot()
    no_match = set()
    print(len(ensembl_to_uniprot_dict))

    # first extract the drug-target pairs from the opentargets json files
    json_files = get_jsonl_files(input_dir)
    for json_file in tqdm(json_files, desc="Processing files"):
        # log.info(json_file)
        for drug_id, target_id in extract_data_from_jsonl(json_file):
            try:
                known_drug_targets.append(
                    {
                        "drug": f"CHEMBL.COMPOUND:{drug_id}",
                        "target": ensembl_to_uniprot_dict[target_id],
                    }
                )
            except:
                no_match.add(target_id)

    log.info(f"No UniProt match for {len(no_match)} targets, e.g. {' ,'.join(list(no_match))}")

    df_known_dt = pd.DataFrame(known_drug_targets)
    print(df_known_dt)
    # TODO: add TrainingConfig
    scores = compute_and_train(df_known_dt, out_dir)


def train_drugbank():
    df_known_dt = "data/drugbank/DB_DTI_4vectordb.csv"
    out_dir = "data/drugbank"

    df = pd.read_csv(df_known_dt)
    convert_dict = get_pref_ids(df["drug"].values, ["PUBCHEM.COMPOUND"])
    print(convert_dict)
    df["drug"] = df["drug"].apply(lambda curie: convert_dict[curie])
    print(df)
    scores = compute_and_train(df, out_dir)

if __name__ == "__main__":
    train_drugbank()
    # train_opentargets("data/download/opentargets/knownDrugsAggregated", "data/opentargets")
