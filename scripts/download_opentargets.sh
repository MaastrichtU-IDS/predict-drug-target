# wget --recursive --no-parent --no-host-directories -P ./data/download/opentargets/ --cut-dirs 8 ftp://ftp.ebi.ac.uk/pub/databases/opentargets/platform/23.09/output/etl/json/knownDrugsAggregated

wget -r -np -nH --cut-dirs=9 -P ./data/download/opentargets/knownDrugsAggregated -e robots=off -R "index.html*" https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/23.09/output/etl/json/knownDrugsAggregated

# mv data/opentarget data/download/targets/knownDrugsAggregated

# Mechanisms of action:
# wget -r -np -nH --cut-dirs=9  -e robots=off -R "index.html*" https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/23.09/output/etl/json/mechanismOfAction/