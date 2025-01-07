from ctypes import alignment

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


def letterEnrichment(kmerLen, sequence, alphabet):
    """
    enrichment = enrichment / np.max(enrichment)
    compared to background positional frequency matrix

    Returns:
    numpy.ndarray: of len(sequence)
    """
    e = 10**-6
    enrichment = []
    _len = 1 + len(sequence) - kmerLen
    noise = positionFreqNoise(kmerLen, sequence, alphabet)
    for i in range(_len):
        slider = sequence[i : i + kmerLen]
        if slider == "TACAT":
            pass
        fold = kmerToFreq(slider, alphabet) / (noise - e)
        enrichment.append(np.sum(fold))
        pass
    enrichment = enrichment / np.max(enrichment)
    return enrichment


def kmerEnrichment():
    return


def getEnrichment():
    return


def kmerHits(kmerLen, sequence, alphabet, threshold=0.51):
    """
    kmer hits for a single sequence above the enrichment threshold
    compared to background positional frequency matrix

    Params:
    threshold: fold enrichment as a fraction of the most fold-enriched kmer (1.0)
    """
    result = {}
    _len = 1 + len(sequence) - kmerLen
    enrichment = letterEnrichment(kmerLen, sequence, alphabet)
    for i in range(_len):
        slider = sequence[i : i + kmerLen]
        if enrichment[i] >= threshold:
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


def kmerConvolution(x, y, alphabet):
    """
    align 2 same length kmer frequency matrices through convolution
    x is expanded, while y is slided over x like a convolutional kernel

    Params:
    op: sum or dot the freq matrices
    """
    x = x[::-1]
    if len(x) != len(y):
        raise ValueError("kmerPairAlign different x and y lengths")
    xFreq = kmerToFreq(x, alphabet)
    zeros = np.zeros(xFreq.shape)
    kmerLen = xFreq.shape[-1]
    xFreq = np.concat((zeros, xFreq, zeros), axis=-1).T
    yFreq = kmerToFreq(y, alphabet).T
    _len = xFreq.shape[0] - kmerLen
    aligns = []
    for i in range(_len):
        slider = xFreq[i : i + kmerLen]
        freq = (slider + yFreq) + (slider * yFreq)
        aligns.append(freq)

    sums = []
    for i in aligns:
        sums.append(np.sum(i))

    _max = max(sums)
    count = 0
    result = np.array([])
    for i in sums:
        if i >= _max:
            result = aligns[count].T
        count += 1
    return result


def kmerAlign(x, y, alphabet):
    a = kmerConvolution(x, y, alphabet)
    b = kmerConvolution(x[::-1], y, alphabet)
    return a if np.sum(a) >= np.sum(b) else b


def groupKmers(kmerLen, sequences, alphabet, threshold=0.51, similarity=0.5):
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


# testSTR1 = "TTTTTACATTTTACATTTTTT"
# # testSTR1 = "TTTTTACGTATTACAT"
# testSTR2 = testSTR1.replace("T", "G")
# L = (testSTR1, testSTR1)
# # y = kmerHits(4, testSTR1, testDNA, threshold=1)
# # print(y)
# # x = getSeqsHits(5, L, alphabet=testDNA)
# # x = groupKmers(5, L, alphabet=testDNA, threshold=1)
# # print(x)
# # x = tuple(x.values())
# # x = np.sum(x, axis=0)
# # print(x)
# # print(freqToSeq(x, testDNA))
# x = groupKmers(5, L, alphabet=testDNA, threshold=0.9)
# print(x.keys())
# print(x)
# print()
# for i in tuple(x.values()):
#     print(i)
#     print(freqToSeq(i, testDNA))
#     print()
s1 = "TTTACATTTAGA"
s2 = s1.replace("T", "G")[::-1]
rs = kmerAlign(s1, s2, testDNA)
print(rs)
print(np.sum(rs))
print(freqToSeq(rs, testDNA))
