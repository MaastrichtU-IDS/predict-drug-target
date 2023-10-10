import logging
from pubchempy import Compound

import requests

## Instantiate logging utility
log = logging.getLogger()
log.propagate = False
log.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s: [%(module)s:%(funcName)s] %(message)s")
console_handler.setFormatter(formatter)
log.addHandler(console_handler)

BOLD = "\033[1m"
END = "\033[0m"
RED = "\033[91m"
YELLOW = "\033[33m"
CYAN = "\033[36m"


def get_smiles_for_drug(drug_id: str):
    # Not all molecule have smiles https://www.ebi.ac.uk/chembl/api/data/molecule/CHEMBL535?format=json
    if drug_id.lower().startswith("chembl.compound:"):
        drug_id = drug_id[len("chembl.compound:"):]
        res = requests.get(f"https://www.ebi.ac.uk/chembl/api/data/molecule/{drug_id}?format=json").json()
        return res["molecule_structures"]["canonical_smiles"]
    if drug_id.lower().startswith("pubchem.compound:"):
        drug_id = drug_id[len("pubchem.compound:"):]
        comp = Compound.from_cid(drug_id)
        return comp.canonical_smiles


def get_seq_for_target(ensembl_id: str):
    # https://www.ebi.ac.uk/proteins/api/proteins/Ensembl:ENSP00000351276?offset=0&size=100&format=json
    if ensembl_id.lower().startswith("ensembl:"):
        ensembl_id = ensembl_id[8:]
    res = requests.get(
        f"https://www.ebi.ac.uk/proteins/api/proteins/Ensembl:{ensembl_id}?offset=0&size=100&format=json"
    ).json()
    return res[0]["sequence"]["sequence"]



def normalize_id_to_translator(ids_list: list):
    """Use Translator SRI NodeNormalization API to get the preferred Translator ID
    for an ID https://nodenormalization-sri.renci.org/docs
    """
    print(len(ids_list))
    converted_ids_obj = {}
    resolve_curies = requests.post(
        "https://nodenormalization-sri.renci.org/get_normalized_nodes",
        json={
            "curies": ids_list,
            "conflate": True,
            "description": False,
            "drug_chemical_conflate": False
        },
        headers={
            "accept": "application/json",
            "Content-Type": "application/json"
        },
        timeout=60,
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
