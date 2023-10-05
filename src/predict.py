import numpy as np
import pandas as pd
import os
import pickle
import requests
from datetime import datetime
from itertools import product

import torch
import esm
from src.vectordb import QdrantDB
from src.utils import log, BOLD, END

recreate_vectordb = False
qdrant_url = "qdrant.137.120.31.148.nip.io"
qdrant_apikey = "TOCHANGE"
collections = [
    {"name": "drug", "size": 512},
    {"name": "target", "size": 1280},
] # Total 1792 features cols
vectordb = QdrantDB(collections=collections, recreate=recreate_vectordb, host=qdrant_url, port=443, api_key=qdrant_apikey)


def load_model(path: str = "models/drug_target.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_smiles_for_drug(chembl_id: str):
    # Not all molevule have smiles https://www.ebi.ac.uk/chembl/api/data/molecule/CHEMBL535?format=json
    if chembl_id.lower().startswith("chembl"):
        chembl_id = chembl_id[7:]
    res = requests.get(f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}?format=json").json()
    return res["molecule_structures"]["canonical_smiles"]

def get_seq_for_target(ensembl_id: str):
    # https://www.ebi.ac.uk/proteins/api/proteins/Ensembl:ENSP00000351276?offset=0&size=100&format=json
    if ensembl_id.lower().startswith("ensembl"):
        ensembl_id = ensembl_id[8:]
    res = requests.get(f"https://www.ebi.ac.uk/proteins/api/proteins/Ensembl:{ensembl_id}?offset=0&size=100&format=json").json()
    return res[0]["sequence"]["sequence"]


def compute_drug_embedding(drugs: list[str]) -> pd.DataFrame:
    embeddings = []
    os.makedirs("tmp", exist_ok=True)
    os.chdir("MolecularTransformerEmbeddings")
    # Get all drugs: print(vectordb.get("drug", None))
    for chembl_id in drugs:
        from_vectordb = vectordb.get("drug", chembl_id)
        if len(from_vectordb) > 0:
            log.info(f"♻️ Drug {from_vectordb[0].payload['id']} retrieved from VectorDB")
            embeddings.append(from_vectordb[0].vector)
            continue
        log.info(f"⏳💊 Drug {chembl_id} not found in VectorDB, computing its embeddings")
        drug_smiles = get_smiles_for_drug(chembl_id)
        print(f"drug_smiles! {chembl_id} {drug_smiles}")
        with open(f"../tmp/drug_smiles.txt", 'w') as f:
            f.write(drug_smiles)
        os.system(f"python embed.py --data_path=../tmp/drug_smiles.txt")
        o = np.load(f"embeddings/drug_smiles.npz")
        files = o.files  # 1 file
        gen_embeddings = []
        for file in files:
            gen_embeddings.append(o[file]) # 'numpy.ndarray' n length x 512
        vectors = np.stack([emb.mean(axis=0) for emb in gen_embeddings])
        vectordb.add("drug", chembl_id, list(vectors)[0], drug_smiles)
        embeddings.append(list(vectors)[0])
    return pd.DataFrame(embeddings)


def compute_target_embedding(targets: list[str]) -> pd.DataFrame:
    embeddings = []
    for ensembl_id in targets:
        from_vectordb = vectordb.get("target", ensembl_id)
        if len(from_vectordb) > 0:
            log.info(f"♻️ Target {from_vectordb[0].payload['id']} retrieved from VectorDB")
            embeddings.append(from_vectordb[0].vector)
            continue
        log.info(f"⏳🎯 Target {ensembl_id} not found in VectorDB, computing its embeddings")
        # TODO: perform bulk compute when multiple embeddings are not cached
        target_seq = get_seq_for_target(ensembl_id)
        # Load ESM-2 model
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disables dropout for deterministic results
        data = [
            (ensembl_id, target_seq),
            # ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

        target_embeddings = torch.stack(sequence_representations,dim=0).numpy() # numpy.ndarray 3775 x 1280
        vectordb.add("target", ensembl_id, list(target_embeddings)[0], target_seq)
        embeddings.append(list(target_embeddings)[0])
    return pd.DataFrame(embeddings)


def get_predictions(drugs: list[str], targets: list[str]):
    time_start = datetime.now()
    model = load_model()

    # Compute embeddings for drugs and target, based on their smiles and amino acid sequence
    drug_embed = compute_drug_embedding(drugs)
    target_embed = compute_target_embedding(targets)

    # Merge embeddings
    merged_embeddings = pd.merge(drug_embed, target_embed, how='cross')
    merged_embeddings.columns = merged_embeddings.columns.astype(str)
    # log.info("Merged embeddings should have 1792 columns (512 from drugs + 1280 from targets)")
    # log.info(merged_embeddings)

    # Get predicted score
    predicted_proba = model.predict_proba(merged_embeddings)
    interaction_scores = predicted_proba[:, 1]  # Probability of class 1

    scores = []
    for drug in drugs:
        for target in targets:
            scores.append({
                "drug": drug,
                "target": target,
                "score": interaction_scores[len(scores)]
            })

    log.info(f"⚡ Interaction score between {drugs} and {targets} is {BOLD}{interaction_scores}{END} (computed in {datetime.now() - time_start})")
    return scores

drugs = ["CHEMBL:CHEMBL535", "CHEMBL:CHEMBL64545"]
targets = ["ENSEMBL:ENSP00000351276", "ENSEMBL:ENSP00000310301"]
# TODO: Pass None to get all targets or all drugs
predictions = get_predictions(drugs, targets)
print(predictions)
