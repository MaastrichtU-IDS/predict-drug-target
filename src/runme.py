from Bio import SeqIO
import subprocess
import os

fasta_file = 'data/drugbank_targets.fasta'
sequences = SeqIO.parse(open(fasta_file),'fasta')
seqs = []
index = 1
for seq in sequences:
    
    if index %101 == 0:
        
        # write the seq
        output_file = f"data/targets.fasta"
        SeqIO.write(seqs, output_file, "fasta")

        # now run esm
        cmd = f"python src/extract.py esm2_t33_650M_UR50D {output_file} {output_file}_emb_esm2 --repr_layers 0 32 33 --include mean per_tok"
        print(os.popen(cmd).read())

        seqs = []
        index = 1
    else:
        seqs.append(seq)
        index += 1
    
print('last batch')
output_file = f"data/targets.fasta"
SeqIO.write(seqs, output_file, "fasta")

# now run esm
#output_file = 'data/few_proteins.fasta'
cmd = f"python src/extract.py esm2_t33_650M_UR50D {output_file} {output_file}_emb_esm2 --repr_layers 0 32 33 --include mean per_tok"
print(os.popen(cmd).read())