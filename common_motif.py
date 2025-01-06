import numpy as np
from utils import getTestDNAStr, hitUp, kmersSetTuple, testDNA


def kmerToFreq(kmer, alphabet):
    """
    Params:
    kmer: str

    Returns:
    numpy.ndarray: Positional Frequency Matrix
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
    numpy.ndarray: Positional Frequency Matrix
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
    threshold: fold enrichment as a fraction of the most fold-enriched sequence (1)
    """
    result = {}
    _len = 1 + len(sequence) - kmerLen
    enrichment = kmerEnrichment(kmerLen, sequence, alphabet)
    for i in range(_len):
        slider = sequence[i : i + kmerLen]
        if enrichment[i] > threshold:
            hitUp(slider, result)
    return result


def seqsHits(kmerLen, sequences):
    """
    kmer hits per sequence for several sequences

    Params:
    sequences: list/tuple of strings

    Returns:
    list: list of dict {'kmers' : count} per input sequence
    """
    result = []
    for i in sequences:
        result.append(kmerHits(kmerLen, i))
    return result


def groupKmers(seqsHits, alphabet, similarity=0.7):
    """
    Params:
    seqsHits: list(dict({'str':int}))
    similarity: fraction of letters shared to group together

    Returns:
    np.array: Positional Frequency Matrix
    """
    kmersIter = kmersSetTuple(seqsHits)

    rows = len(alphabet)
    cols = len(kmersIter[0])
    zeros = np.zeros((rows, cols))
    result = {}
    for i in kmersIter:
        result.update({i: zeros})

    for _dict in seqsHits:
        for i in tuple(_dict.keys()):
            for j in kmersIter:
                if pairKmers(i, j, alphabet, similarity):
                    result[j] += kmerToFreq(i, alphabet) * _dict[i]
    return result


testSTR1 = "TTTTTACGTATTTTT"
testSTR2 = testSTR1.replace("T", "G")
L = (testSTR1, testSTR1)
y = kmerHits(5, testSTR1, testDNA, threshold=0.9)
print(y)
# x = seqsHits(5, L)
# x = groupKmers()
# print(x)
