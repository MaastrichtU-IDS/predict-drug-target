import logging

import requests
import pubchempy as pcp

VECTORDB_MAX_LIMIT = 100000

EMBEDDINGS_SIZE_DRUG = 512
EMBEDDINGS_SIZE_TARGET = 1280
COLLECTIONS = [
    {"name": "drug", "size": EMBEDDINGS_SIZE_DRUG},
    {"name": "target", "size": EMBEDDINGS_SIZE_TARGET},
]  # Total 1792 features cols


## Instantiate logging utility
log = logging.getLogger(__name__)
log.propagate = False
log.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s: [%(module)s:%(funcName)s] %(message)s")
console_handler.setFormatter(formatter)
log.addHandler(console_handler)

# Silence qdrant log.infos
default_logger = logging.getLogger()
default_logger.setLevel(logging.WARNING)


BOLD = "\033[1m"
END = "\033[0m"
RED = "\033[91m"
YELLOW = "\033[33m"
CYAN = "\033[36m"

TIMEOUT = 30


def get_smiles_for_drug(drug_id: str):
    # Not all molecule have smiles https://www.ebi.ac.uk/chembl/api/data/molecule/CHEMBL4297578?format=json
    # CHEMBL.COMPOUND:CHEMBL4297578
    if drug_id.lower().startswith("chembl.compound:"):
        drug_id = drug_id[len("chembl.compound:") :]
        res = requests.get(
            f"https://www.ebi.ac.uk/chembl/api/data/molecule/{drug_id}?format=json", timeout=TIMEOUT
        ).json()
        # log.info(f'{drug_id} | {res["molecule_structures"]["canonical_smiles"]} | {res["pref_name"]}')
        return res["molecule_structures"]["canonical_smiles"], res["pref_name"]
    if drug_id.lower().startswith("pubchem.compound:"):
        drug_id = drug_id[len("pubchem.compound:") :]
        comp = pcp.Compound.from_cid(drug_id)
        return comp.canonical_smiles, comp.iupac_name


def get_seq_for_target(target_id: str):
    # https://www.ebi.ac.uk/proteins/api/proteins/Ensembl:ENSP00000351276?offset=0&size=100&format=json
    if target_id.lower().startswith("ensembl:"):
        target_id = target_id[len("ensembl:") :]
        res = requests.get(
            f"https://www.ebi.ac.uk/proteins/api/proteins/Ensembl:{target_id}?offset=0&size=100&format=json",
            timeout=TIMEOUT,
        ).json()
        return res[0]["sequence"]["sequence"], res[0]["protein"]["recommendedName"]["fullName"]["value"]
    if target_id.lower().startswith("uniprotkb:"):
        target_id = target_id[len("uniprotkb:") :]
        res = requests.get(f"https://rest.uniprot.org/uniprotkb/{target_id}?format=json", timeout=TIMEOUT).json()
        return res["sequence"]["value"], res["proteinDescription"]["recommendedName"]["fullName"]["value"]


def normalize_id_to_translator(ids_list: list):
    """Use Translator SRI NodeNormalization API to get the preferred Translator ID
    for an ID https://nodenormalization-sri.renci.org/docs
    """
    print(len(ids_list))
    converted_ids_obj = {}
    resolve_curies = requests.post(
        "https://nodenormalization-sri.renci.org/get_normalized_nodes",
        json={"curies": ids_list, "conflate": True, "description": False, "drug_chemical_conflate": False},
        headers={"accept": "application/json", "Content-Type": "application/json"},
        timeout=TIMEOUT,
    )
    resolve_curies.raise_for_status()
    # Get corresponding OMIM IDs for MONDO IDs if match
    resp = resolve_curies.json()
    # print(resp)
    for converted_id, translator_ids in resp.items():
        try:
            pref_id = translator_ids["id"]["identifier"]
            log.debug(converted_id + " > " + pref_id)
            converted_ids_obj[converted_id] = pref_id
        except Exception:
            log.debug("❌️ " + converted_id + " > " + str(translator_ids))
            converted_ids_obj[converted_id] = None

    return converted_ids_obj
