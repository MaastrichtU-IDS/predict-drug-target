import csv
from io import StringIO
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

filename = 'data/targets.csv'
seqs = []

with open(filename,'r') as data:
   for line in csv.DictReader(data):
        fasta_io = StringIO(line['seq']) 
        records = SeqIO.parse(fasta_io, "fasta")
        
        for rec in records:
            r = SeqRecord(
                Seq(rec.seq),
                id=line['target_id'],
                name='',
                description='')
            seqs.append(r)
        
        fasta_io.close()
           # print(r)
SeqIO.write(seqs, "data/drugbank_targets.fasta", "fasta")
