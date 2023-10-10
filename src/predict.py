import os
import pickle
from datetime import datetime

import esm
import numpy as np
import pandas as pd
import torch

from src.utils import BOLD, END, log, get_smiles_for_drug, get_seq_for_target
from src.vectordb import init_vectordb, VectorDB


def load_model(path: str = "models/drug_target.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_drug_embedding(drugs: list[str], vectordb: VectorDB) -> pd.DataFrame:
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
        with open("../tmp/drug_smiles.txt", "w") as f:
            f.write(drug_smiles)
        os.system("python embed.py --data_path=../tmp/drug_smiles.txt")
        o = np.load("embeddings/drug_smiles.npz")
        files = o.files  # 1 file
        gen_embeddings = []
        for file in files:
            gen_embeddings.append(o[file])  # 'numpy.ndarray' n length x 512
        vectors = np.stack([emb.mean(axis=0) for emb in gen_embeddings])
        # In this case we vectorize one by one, so only 1 row in the array
        vector = vectors[0].tolist()
        vectordb.add("drug", chembl_id, vector, drug_smiles)
        embeddings.append(list(vector))
    return pd.DataFrame(embeddings)


def compute_target_embedding(targets: list[str], vectordb: VectorDB) -> pd.DataFrame:
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

        target_embeddings = torch.stack(sequence_representations, dim=0).numpy()  # numpy.ndarray 3775 x 1280
        vector = target_embeddings[0].tolist()
        vectordb.add("target", ensembl_id, vector, target_seq)
        embeddings.append(next(iter(target_embeddings)))  # get 0 of the list
    return pd.DataFrame(embeddings)


def get_predictions(drugs: list[str], targets: list[str]):
    time_start = datetime.now()
    model = load_model()

    # Compute embeddings for drugs and target, based on their smiles and amino acid sequence
    drug_embed = compute_drug_embedding(drugs)
    target_embed = compute_target_embedding(targets)

    # Merge embeddings
    merged_embeddings = pd.merge(drug_embed, target_embed, how="cross")
    merged_embeddings.columns = merged_embeddings.columns.astype(str)
    # log.info("Merged embeddings should have 1792 columns (512 from drugs + 1280 from targets)")
    # log.info(merged_embeddings)

    # Get predicted score
    predicted_proba = model.predict_proba(merged_embeddings)
    interaction_scores = predicted_proba[:, 1]  # Probability of class 1

    scores = []
    for drug in drugs:
        for target in targets:
            scores.append({"drug": drug, "target": target, "score": interaction_scores[len(scores)]})

    log.info(
        f"⚡ Interaction score between {drugs} and {targets} is {BOLD}{interaction_scores}{END} (computed in {datetime.now() - time_start})"
    )
    return scores


if __name__ == "__main__":
    vectordb = init_vectordb(recreate=False)
    # drugs = ["CHEMBL.COMPOUND:CHEMBL535", "CHEMBL.COMPOUND:CHEMBL64545"]
    drugs = ["PUBCHEM.COMPOUND:5329102", "PUBCHEM.COMPOUND:4039"]
    targets = ["ENSEMBL:ENSP00000351276", "ENSEMBL:ENSP00000310301"]
    # TODO: Pass None to get all targets or all drugs
    predictions = get_predictions(drugs, targets)
    print(predictions)
