import logging

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


def normalize_id_to_translator(ids_list):
    """Use Translator SRI NodeNormalization API to get the preferred Translator ID
    for an ID https://nodenormalization-sri.renci.org/docs
    """
    print(len(ids_list))
    converted_ids_obj = {}
    resolve_curies = requests.post(
        "https://nodenormalization-sri.renci.org/get_normalized_nodes",
        data={"curies": ids_list[:100]},
        timeout=60,
    )
    # print(resolve_curies)
    resolve_curies.raise_for_status()
    # Get corresponding OMIM IDs for MONDO IDs if match
    resp = resolve_curies.json()
    # print(resp)
    for converted_id, translator_ids in resp.items():
        try:
            pref_id = translator_ids["id"]["identifier"]
            log.info(converted_id + " > " + pref_id)
            converted_ids_obj[converted_id] = pref_id
        except Exception:
            log.error("❌️ " + converted_id + " > " + str(translator_ids))

    return converted_ids_obj
