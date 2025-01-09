# Common Motif Discovery

python tool to find a common DNA, RNA or amino acid subsequence common to several sequences

## Algorithm description
- get every kmer of base length (5)
- get kmers common to all sequences by similarity
- group the common kmers by similarity
- get motifs from each group
- repeat to grow the obtanied motifs

### Details
- Similarity = matches / length, done by multipying matrices then dividing reduce sum by length