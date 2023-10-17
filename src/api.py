import logging

from trapi_predict_kit import TRAPI, settings

from src.predict import get_drug_target_predictions

log_level = logging.INFO
logging.basicConfig(level=log_level)


trapi_example = {
    "message": {
        "query_graph": {
            "edges": {"e01": {"object": "n1", "predicates": ["biolink:interacts_with"], "subject": "n0"}},
            "nodes": {
                "n0": {"categories": ["biolink:Drug"],
                    "ids": ["PUBCHEM.COMPOUND:5329102", "PUBCHEM.COMPOUND:4039"]
                },
                "n1": {
                    "categories": ["biolink:Protein"],
                    "ids": ["ENSEMBL:ENSP00000351276", "ENSEMBL:ENSP00000310301"],
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
    "termsOfService": "https://github.com/your-org-or-username/my-model/blob/main/LICENSE.txt",
    "x-translator": {
        "component": "KP",
        # TODO: update the Translator team to yours
        "team": ["Clinical Data Provider"],
        "biolink-version": settings.BIOLINK_VERSION,
        "infores": "infores:openpredict",
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
