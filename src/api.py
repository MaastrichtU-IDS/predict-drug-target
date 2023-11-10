import logging

from trapi_predict_kit import TRAPI, settings

from src.predict import get_drug_target_predictions
from src.train import train
from src.utils import COLLECTIONS
from src.vectordb import init_vectordb

log_level = logging.INFO
logging.basicConfig(level=log_level)


trapi_example = {
    "message": {
        "query_graph": {
            "edges": {"e01": {"object": "n1", "predicates": ["biolink:interacts_with"], "subject": "n0"}},
            "nodes": {
                "n0": {"categories": ["biolink:Drug"], "ids": ["PUBCHEM.COMPOUND:5329102", "PUBCHEM.COMPOUND:4039", 'CHEMBL.COMPOUND:CHEMBL1431']},
                "n1": {
                    "categories": ["biolink:Protein"],
                    "ids": ["UniProtKB:O75251"],
                },
            },
        }
    },
    "query_options": {"max_score": 1, "min_score": 0.1, "n_results": 10},
}


openapi_info = {
    "contact": {
        "name": "Firstname Lastname",
        "email": "email@example.com",
        # "x-id": "https://orcid.org/0000-0000-0000-0000",
        "x-role": "responsible developer",
    },
    "license": {
        "name": "MIT license",
        "url": "https://opensource.org/licenses/MIT",
    },
    "termsOfService": "https://github.com/MaastrichtU-IDS/predict-drug-target/blob/main/LICENSE.txt",
    "x-translator": {
        "component": "KP",
        "team": ["Clinical Data Provider"],
        "biolink-version": settings.BIOLINK_VERSION,
        "infores": "infores:predict-drug-target",
        "externalDocs": {
            "description": "The values for component and team are restricted according to this external JSON schema. See schema and examples at url",
            "url": "https://github.com/NCATSTranslator/translator_extensions/blob/production/x-translator/",
        },
    },
    "x-trapi": {
        "version": settings.TRAPI_VERSION,
        "asyncquery": False,
        "operations": [
            "lookup",
        ],
        "externalDocs": {
            "description": "The values for version are restricted according to the regex in this external JSON schema. See schema and examples at url",
            "url": "https://github.com/NCATSTranslator/translator_extensions/blob/production/x-trapi/",
        },
    },
}

app = TRAPI(
    predict_endpoints=[get_drug_target_predictions],
    info=openapi_info,
    title="Predict Drug Target interactions TRAPI",
    version="1.0.0",
    openapi_version="3.0.1",
    description="""Get predicted protein targets for a given drug
\n\nService supported by the [NCATS Translator project](https://ncats.nih.gov/translator/about)""",
    itrb_url_prefix="predict-drug-target",
    dev_server_url="https://predict-drug-target.137.120.31.160.nip.io",
    trapi_example=trapi_example,
    # trapi_description=""
)


@app.post("/reset-vectordb", name="Reset vector database", description="Reset the collections in the vectordb")
def post_reset_vectordb(api_key: str):
    init_vectordb(recreate=True, api_key=api_key)
    return {"status": "ok"}


@app.post("/train", name="Run training", description="Run training of the model")
def post_train(api_key: str):
    # init_vectordb(recreate=True, api_key=api_key)
    scores = train()
    return scores
