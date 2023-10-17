#!/usr/bin/bash
# pip install git+https://github.com/facebookresearch/esm.git

# Get drug-target data from the Bio2RDF SPARQL endpoint
./get_bio2rdf_data.sh

# Generate drugs embeddings
esm-extract esm2_t33_650M_UR50D data/download/drugbank_targets.fasta data/vectors/drugbank_targets_esm2_l33_mean --repr_layers 33 --include mean

# Install the Molecular Transformer Embeddings for proteins
# https://github.com/mpcrlab/MolecularTransformerEmbeddings
git clone https://github.com/mpcrlab/MolecularTransformerEmbeddings.git
cd MolecularTransformerEmbeddings
chmod +x download.sh
./download.sh
python embed.py --data_path=../data/download/drugbank_smiles.txt
mv embeddings/drugbank_smiles.npz ../data/vectors/
cd ..
