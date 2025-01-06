import numpy as np
from utils import freqToSeq, getTestDNAStr, hitUp, kmersSetTuple, testDNA


def kmerToFreq(kmer, alphabet):
    """
    Params:
    kmer: str

    Returns:
    numpy.ndarray: positional frequency Matrix
    """
    matrix = []
    for i in alphabet:
        row = []
        for j in kmer:
            if i == j:
                row.append(1)
            else:
                row.append(0)
        matrix.append(row)
    return np.array(matrix)


def pairKmers(seq1, seq2, alphabet, similarity=0.7):
    """
    Params:
    similarity: fraction of letters shared to group together

    Returns:
    bool: whether seq1 and seq2 pair
    """
    similarity = len(seq1) * similarity
    x = kmerToFreq(seq1, alphabet)
    y = kmerToFreq(seq2, alphabet)
    score = np.sum(x * y)
    return score >= similarity


def positionFreqNoise(kmerLen, sequence, alphabet):
    """
    sum of letter frequency for every kmer of kmerLen

    Returns:
    numpy.ndarray: positional frequency matrix
    """
    result = np.zeros((len(alphabet), kmerLen))
    _len = 1 + len(sequence) - kmerLen
    for i in range(_len):
        slider = sequence[i : i + kmerLen]
        result += kmerToFreq(slider, alphabet)
    result = (result / np.sum(result)) * kmerLen
    return result


def kmerEnrichment(kmerLen, sequence, alphabet):
    """
    enrichment = enrichment / np.max(enrichment)

    Returns:
    numpy.ndarray: of len(sequence)
    """
    enrichment = []
    _len = 1 + len(sequence) - kmerLen
    noise = positionFreqNoise(kmerLen, sequence, alphabet)
    for i in range(_len):
        slider = sequence[i : i + kmerLen]
        fold = kmerToFreq(slider, alphabet) / noise
        enrichment.append(np.sum(fold))
    enrichment = enrichment / np.max(enrichment)
    return enrichment


def kmerHits(kmerLen, sequence, alphabet, threshold=0.51):
    """
    kmer hits for a single sequence above the enrichment threshold
    compared to background positional frequency matrix

    Params:
    threshold: fold enrichment as a fraction of the most fold-enriched kmer (1.0)
    """
    result = {}
    _len = 1 + len(sequence) - kmerLen
    enrichment = kmerEnrichment(kmerLen, sequence, alphabet)
    for i in range(_len):
        slider = sequence[i : i + kmerLen]
        if enrichment[i] > threshold:
            hitUp(slider, result)
    return result


def seqsHits(kmerLen, sequences, alphabet, threshold=0.51):
    """
    kmer hits per sequence for several sequences

    Params:
    sequences: list/tuple of strings
    threshold: kmer fold enrichment as a fraction of the most fold-enriched  kmer (1.0) per seq

    Returns:
    list: list of dict {'kmers' : count} per input sequence
    """
    result = []
    for i in sequences:
        result.append(kmerHits(kmerLen, i, alphabet, threshold))
    return result


def groupKmers(kmerLen, sequences, alphabet, threshold=0.51, similarity=0.7):
    """
    Params:
    similarity: fraction of letters shared to group together
    threshold: kmer fold enrichment as a fraction of the most fold-enriched  kmer (1.0) per seq

    Returns:
    dict: each kmer and the kmers it groups with as a positional freq matrix {str : ndarray}
    """
    seqsHitsIter = seqsHits(kmerLen, sequences, alphabet, threshold=threshold)
    kmersIter = kmersSetTuple(seqsHitsIter)

    rows = len(alphabet)
    cols = len(kmersIter[0])
    zeros = np.zeros((rows, cols))
    result = {}
    for i in kmersIter:
        result.update({i: zeros})

    for _dict in seqsHitsIter:
        for i in tuple(_dict.keys()):
            for j in kmersIter:
                if pairKmers(i, j, alphabet, similarity):
                    result[j] += kmerToFreq(i, alphabet) * _dict[i]
    return result


testSTR1 = "TTTTTACGTATTTTT"
testSTR2 = testSTR1.replace("T", "G")
L = (testSTR1, testSTR1)
# y = kmerHits(5, testSTR1, testDNA, threshold=0.9)
# print(y)
# x = getSeqsHits(5, L, alphabet=testDNA)
x = groupKmers(5, L, alphabet=testDNA, threshold=0.7)
print(x)
x = tuple(x.values())
x = np.sum(x, axis=0)
# print(x)
print(freqToSeq(x, testDNA))
print(freqToSeq(x, testDNA, skip_draw=True))
