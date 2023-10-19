
import esm
import pandas as pd
import torch
from smiles_transformer import get_smiles_embeddings
from tqdm import tqdm

from src.utils import (
    COLLECTIONS,
    EMBEDDINGS_SIZE_DRUG,
    EMBEDDINGS_SIZE_TARGET,
    VECTORDB_MAX_LIMIT,
    get_seq_for_target,
    get_smiles_for_drug,
    log,
)
from src.vectordb import VectorDB, init_vectordb

VECTORDB = init_vectordb(COLLECTIONS, recreate=False)


def compute_drug_embedding(
    vectordb: VectorDB, drugs: list[str] | None = None, length: int = EMBEDDINGS_SIZE_DRUG
) -> pd.DataFrame:
    """Compute embeddings for a list of drug ID based on its retrieved SMILES.
    Returns a pandas dataframe with a "drug" column containing the drug ID,
    and all other columns contains the embeddings
    """
    df = pd.DataFrame(columns=["drug", *list(range(length))])
    # If no drug provided we get all drugs in vectordb
    if not drugs:
        # Get all drugs takes ~10s for 5k5 drugs
        drugs_list = vectordb.get("drug", None, limit=VECTORDB_MAX_LIMIT)
        log.info(f"Retrieved {len(drugs_list)} drugs")
        drugs_list = [{"drug": drug.payload["id"], **dict(enumerate(drug.vector, 1))} for drug in drugs_list]
        df = pd.DataFrame.from_records(drugs_list)
        return df

    # TODO: make sure drugs in the list are unique

    # Otherwise check if drug embedding already in vectordb
    upload_list = []
    list_drugs_no_smiles = []
    drugs_no_embed = {}
    labels_dict = {}
    for drug_id in tqdm(drugs, desc="Check drugs in vector db, or get SMILES"):
        from_vectordb = vectordb.get("drug", drug_id)
        if len(from_vectordb) > 0:
            log.info(f"♻️ Drug {from_vectordb[0].payload['id']} retrieved from VectorDB")
            embeddings = from_vectordb[0].vector
            embeddings.insert(0, drug_id)
            # df = pd.concat([df, pd.DataFrame(embeddings)], ignore_index = True)
            df.loc[len(df)] = embeddings
        else:
            # If not in vectordb we get its smile and add it to the list to compute
            try:
                drug_smiles, drug_label = get_smiles_for_drug(drug_id)
                drugs_no_embed[drug_smiles] = drug_id
                labels_dict[drug_id] = drug_label
            except:
                list_drugs_no_smiles.append(drug_id)

    if list_drugs_no_smiles:
        log.info(f"⚠️ We could not find SMILES for {len(list_drugs_no_smiles)}")
        log.info(list_drugs_no_smiles)
    if not drugs_no_embed:  # No embeddings to generate
        return df

    # Then we compute embeddings for all drugs not in vectordb
    log.info(f"⏳💊 {len(drugs_no_embed)} Drugs not found in VectorDB, computing their embeddings from SMILES")
    embed_dict = get_smiles_embeddings(list(drugs_no_embed.keys()))

    # Finally we add the newly computed embeddings to the vectordb
    for smiles, embeddings in embed_dict.items():
        log.info(len(embeddings))
        drug_id = drugs_no_embed[smiles]
        upload_list.append(
            {"vector": embeddings, "payload": {"id": drug_id, "sequence": smiles, "label": labels_dict[drug_id]}}
        )
        df.loc[len(df)] = [drug_id] + embeddings

    vectordb.add("drug", upload_list)
    return df


def compute_target_embedding(
    vectordb: VectorDB, targets: list[str], length: int = EMBEDDINGS_SIZE_TARGET
) -> pd.DataFrame:
    """Compute embeddings for a list of target ID based on its retrieved amino acid sequence.
    Returns a pandas dataframe with a "target" column containing the target ID,
    and all other columns contains the embeddings
    """
    df = pd.DataFrame(columns=["target", *list(range(length))])
    # If not target provided we get all targets in vectordb
    if not targets:
        targets_list = vectordb.get("target", None, limit=VECTORDB_MAX_LIMIT)
        log.info(f"Retrieved {len(targets_list)} targets")
        targets_list = [
            {"target": target.payload["id"], **dict(enumerate(target.vector, 1))} for target in targets_list
        ]
        df = pd.DataFrame.from_records(targets_list)
        return df

    # Otherwise check if target embedding already in vectordb
    upload_list = []
    targets_no_embed = {}
    list_targets_no_seq = []
    labels_dict = {}
    for target_id in tqdm(targets, desc="Check targets in vector db, or get their AA seq"):
        # Check if we can find it in the vectordb
        from_vectordb = vectordb.get("target", target_id)
        if len(from_vectordb) > 0:
            log.info(f"♻️ Target {from_vectordb[0].payload['id']} retrieved from VectorDB")
            embeddings = from_vectordb[0].vector
            embeddings.insert(0, target_id)
            df.loc[len(df)] = embeddings
        else:
            # If not in vectordb we get its smile and add it to the list to compute
            try:
                target_seq, target_label = get_seq_for_target(target_id)
                targets_no_embed[target_seq] = target_id
                labels_dict[target_id] = target_label
            except:
                list_targets_no_seq.append(target_id)

    if list_targets_no_seq:
        log.info(f"⚠️ We could not find AA sequences for {len(list_targets_no_seq)}")
        log.info(list_targets_no_seq)
    if not targets_no_embed:  # No embeddings to generate
        return df

    # Compute the missing targets embeddings
    log.info(f"⏳🎯 {len(targets_no_embed)} targets not found in VectorDB, computing their embeddings")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    data = [(target_id, target_seq) for target_seq, target_id in targets_no_embed.items()]
    # data = [ ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"), ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

    target_embeddings = torch.stack(sequence_representations, dim=0).numpy().tolist()  # numpy.ndarray 3775 x 1280
    out_dict = {seq: matrix for seq, matrix in zip(targets_no_embed.keys(), target_embeddings)}

    # Add the computed embeddings to the vectordb
    for target_seq, embeddings in out_dict.items():
        target_id = targets_no_embed[target_seq]
        upload_list.append(
            {
                "vector": embeddings,
                "payload": {"id": target_id, "sequence": target_seq, "label": labels_dict[target_id]},
            }
        )
        df.loc[len(df)] = [target_id] + embeddings
    vectordb.add("target", upload_list)
    return df
