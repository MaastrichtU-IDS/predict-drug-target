curl -F 'format=text/csv' -F "query=@drug_targets.rq" "https://bio2rdf.org/sparql" > drug_targets.csv
curl -F 'format=text/csv' -F "query=@drugs.rq" "https://bio2rdf.org/sparql" > drugs.csv
curl -F 'format=text/csv' -F "query=@targets.rq" "https://bio2rdf.org/sparql" > targets.csv