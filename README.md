# Common Motif Discovery
Python bioinformatic tool to find a common DNA, RNA or amino acid subsequence common to several sequences.

### Motivation
There's an obvious lack of cell, tissue and stage specific motifs for accurate gene expression.
Intended to make easy to find common substrings motifs across input strings. The only alternative is Homer ( Hypergeometric Optimization of Motif EnRichment ) which is rather stringent, difficult to use and limited to Linux systems.

## Algorithm description
- get every kmer of base length (5)
- get kmers common to all sequences by similarity (matches / length)
- group the common kmers by similarity
- get motifs from each group
- repeat to grow the obtanied motifs