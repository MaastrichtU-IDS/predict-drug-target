# drug-target-emb-predict
This project uses [ESM2](https://github.com/facebookresearch/esm) protein embeddings and [MolecularTransformer](https://github.com/mpcrlab/MolecularTransformerEmbeddings) drug embeddings to train a linear classifier to predict drug-targets.

## Install

Create and activate local environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Install requirements

```bash
pip install -e .
```

Query the Bio2RDF endpoint to get drugs and their smiles, targets and their protein sequences, and the set of known drug-target pairs
```bash
./get_bio2rdf_data.sh
```

Process the Bio2RDF data to generate the inputs needed for the two embeddings methods
```bash
python src/prepare.py
```

Install the ESM library
```bash
pip install git+https://github.com/facebookresearch/esm.git
```

Generate the protein embeddings
```bash
esm-extract esm2_t33_650M_UR50D data/download/drugbank_targets.fasta data/vectors/drugbank_targets_esm2_l33_mean --repr_layers 33 --include mean
```

Install the [Molecular Transformer Embeddings](https://github.com/mpcrlab/MolecularTransformerEmbeddings)
```bash
git clone https://github.com/mpcrlab/MolecularTransformerEmbeddings.git
cd MolecularTransformerEmbeddings
chmod +x download.sh
./download.sh
```
if you get an error (bash: ./download.sh: /bin/bash^M: bad interpreter: No such file or directory) running the download script, then run dos2unix

Generate the drug embeddings
```bash
python embed.py --data_path=../data/download/drugbank_smiles.txt
mv embeddings/drugbank_smiles.npz ../data/vectors/
cd ..
```

Run the prediction tool
```bash
python src/dt_predict.py
```

Results are in results folder

