import os
import pickle
from datetime import datetime

import esm
import numpy as np
import pandas as pd
import torch
from trapi_predict_kit import PredictInput, PredictOutput, trapi_predict

from src.utils import (
    BOLD,
    COLLECTIONS,
    EMBEDDINGS_SIZE_DRUG,
    EMBEDDINGS_SIZE_TARGET,
    END,
    VECTORDB_MAX_LIMIT,
    get_seq_for_target,
    get_smiles_for_drug,
    log,
)
from src.vectordb import VectorDB, init_vectordb

VECTORDB = init_vectordb(COLLECTIONS, recreate=False)


def load_model(path: str = "models/drug_target.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_drug_embedding(
    vectordb: VectorDB, drugs: list[str] | None = None, length: int = EMBEDDINGS_SIZE_DRUG
) -> pd.DataFrame:
    df = pd.DataFrame(columns=["drug", *list(range(length))])
    if not drugs:
        # Get all drugs takes ~10s for 5k5 drugs
        drugs_list = vectordb.get("drug", None, limit=VECTORDB_MAX_LIMIT)
        log.info(f"Retrieved {len(drugs_list)} drugs")
        drugs_list = [{"drug": drug.payload["id"], **dict(enumerate(drug.vector, 1))} for drug in drugs_list]
        df = pd.DataFrame.from_records(drugs_list)
        return df

    os.makedirs("tmp", exist_ok=True)
    os.chdir("MolecularTransformerEmbeddings")
    for drug_id in drugs:
        from_vectordb = vectordb.get("drug", drug_id)
        if len(from_vectordb) > 0:
            log.info(f"♻️ Drug {from_vectordb[0].payload['id']} retrieved from VectorDB")
            embeddings = from_vectordb[0].vector
            embeddings.insert(0, drug_id)
            # df = pd.concat([df, pd.DataFrame(embeddings)], ignore_index = True)
            df.loc[len(df)] = embeddings
            continue

        log.info(f"⏳💊 Drug {drug_id} not found in VectorDB, computing its embeddings")
        drug_smiles = get_smiles_for_drug(drug_id)
        print(f"drug_smiles! {drug_id} {drug_smiles}")
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
        embeddings = vectors[0].tolist()
        vectordb.add("drug", drug_id, embeddings, drug_smiles)
        embeddings.insert(0, drug_id)
        df.loc[len(df)] = embeddings
    os.chdir("..")
    return df


def compute_target_embedding(
    vectordb: VectorDB, targets: list[str], length: int = EMBEDDINGS_SIZE_TARGET
) -> pd.DataFrame:
    df = pd.DataFrame(columns=["target", *list(range(length))])
    if not targets:
        # Get all targets
        targets_list = vectordb.get("target", None, limit=VECTORDB_MAX_LIMIT)
        log.info(f"Retrieved {len(targets_list)} targets")
        targets_list = [
            {"target": target.payload["id"], **dict(enumerate(target.vector, 1))} for target in targets_list
        ]
        df = pd.DataFrame.from_records(targets_list)
        return df

    for target_id in targets:
        # Check if we can find it in the vectordb
        from_vectordb = vectordb.get("target", target_id)
        if len(from_vectordb) > 0:
            log.info(f"♻️ Target {from_vectordb[0].payload['id']} retrieved from VectorDB")
            embeddings = from_vectordb[0].vector
            embeddings.insert(0, target_id)
            df.loc[len(df)] = embeddings
            continue

        log.info(f"⏳🎯 Target {target_id} not found in VectorDB, computing its embeddings")
        # TODO: perform bulk compute when multiple embeddings are not cached
        target_seq = get_seq_for_target(target_id)
        # Load ESM-2 model
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disables dropout for deterministic results
        data = [
            (target_id, target_seq),
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
        embeddings = target_embeddings[0].tolist()
        vectordb.add("target", target_id, embeddings, target_seq)
        embeddings.insert(0, target_id)
        df.loc[len(df)] = embeddings
    return df


@trapi_predict(
    path="/predict-drug-target",
    name="Get predicted score for interactions between drugs and targets (protein)",
    description="Return the predicted targets for a given entity: drug (PubChem ID) or target (UniProtKB ID), with confidence scores.",
    edges=[
        {
            "subject": "biolink:Drug",
            "predicate": "biolink:interacts_with",
            "inverse": "biolink:interacts_with",
            "object": "biolink:Protein",
        },
    ],
    nodes={"biolink:Protein": {"id_prefixes": ["UniProtKB", "ENSEMBL"]}, "biolink:Drug": {"id_prefixes": ["PUBCHEM.COMPOUND"]}},
)
def get_drug_target_predictions(request: PredictInput) -> PredictOutput:
    time_start = datetime.now()
    model = load_model()

    # Compute embeddings for drugs and target, based on their smiles and amino acid sequence
    drug_embed = compute_drug_embedding(VECTORDB, request.subjects)
    target_embed = compute_target_embedding(VECTORDB, request.objects)
    # print("DRUGS TARGETS", drug_embed)
    # print(target_embed)

    # Merge embeddings, results should have 1792 columns (512 from drugs + 1280 from targets)
    df = pd.merge(drug_embed, target_embed, how="cross")
    df.columns = df.columns.astype(str)
    merged_embeddings = df.drop(columns=["drug", "target"])
    merged_embeddings.columns = range(merged_embeddings.shape[1])  # use default column names, same as during training
    # log.info(df)

    # Get predicted score
    predicted_proba = model.predict_proba(merged_embeddings)
    df["score"] = predicted_proba[:, 1]  # Probability of class 1
    df = df.sort_values(by="score", ascending=False)
    df.rename(columns={"drug": "subject", "target": "object"}, inplace=True)
    score_df = df[["subject", "object", "score"]]
    # Convert to list of dicts
    log.info(
        f"⚡ {BOLD}{len(df)}{END} interaction scores computed in {BOLD}{datetime.now() - time_start}{END}\n{score_df.iloc[:10]}"
    )
    scores_list = score_df.to_dict(orient="records")
    return {"hits": scores_list, "count": len(scores_list)}
