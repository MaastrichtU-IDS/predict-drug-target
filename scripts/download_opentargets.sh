# Dont work on UM servers:
# wget --recursive --no-parent --no-host-directories -P ./data/download/opentargets/ --cut-dirs 8 ftp://ftp.ebi.ac.uk/pub/databases/opentargets/platform/23.09/output/etl/json/knownDrugsAggregated

# Works on UM servers, but ddl also other folders alongside knownDrugsAggregated:
wget -r -np -nH --cut-dirs=8 -P ./data/download/opentargets/ -e robots=off -R "index.html*" https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/23.09/output/etl/json/knownDrugsAggregated

# Mechanisms of action:
# wget -r -np -nH --cut-dirs=8 -P ./data/download/opentargets/ -e robots=off -R "index.html*" https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/23.09/output/etl/json/mechanismOfAction/