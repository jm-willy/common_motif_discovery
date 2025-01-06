import numpy as np

from common_motif_discovery.utils import getTestDNAStr, hitUp, testDNA


def kmerToFreq(kmer, alphabet):
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
    similarity = len(seq1) * similarity
    x = kmerToFreq(seq1, alphabet)
    y = kmerToFreq(seq2, alphabet)
    score = np.sum(x * y)
    return score >= similarity


def positionFreqNoise(kmerLen, sequence, alphabet, relative=False):
    result = np.zeros((len(alphabet), kmerLen))
    _len = 1 + len(sequence) - kmerLen
    for i in range(_len):
        slider = sequence[i : i + kmerLen]
        result += kmerToFreq(slider, alphabet)
    if relative:
        result = (result / np.sum(result)) * kmerLen
    return result


def kmerEnrichment(kmerLen, sequence, alphabet):
    enrichment = []
    _len = 1 + len(sequence) - kmerLen
    noise = positionFreqNoise(kmerLen, sequence, alphabet, relative=True)
    for i in range(_len):
        slider = sequence[i : i + kmerLen]
        fold = kmerToFreq(slider, alphabet) / noise
        enrichment.append(np.sum(fold))
    enrichment = enrichment / np.max(enrichment)
    return enrichment


def kmerHits(kmerLen, sequence, alphabet, threshold=0.51):
    """
    kmer hits for a single sequence
    """
    result = {}
    _len = 1 + len(sequence) - kmerLen
    enrichment = kmerEnrichment(kmerLen, sequence, alphabet)
    for i in range(_len):
        slider = sequence[i : i + kmerLen]
        if enrichment[i] > threshold:
            hitUp(slider, result)
    return result


def kmersSetTuple(hitDicts):
    kmersIter = set()
    for _dict in hitDicts:
        for i in tuple(_dict.keys()):
            kmersIter.add(i)
    return tuple(kmersIter)


def seqsHits(minKmerLen, sequences):
    """
    kmer hits for several sequences
    """
    if type(sequences) is not list and type(sequences) is not tuple:
        typesList = [type(i) for i in sequences]
        raise ValueError(
            f"sequences must be a list/tuple of strings. Found:\n{type(typesList), typesList}"
        )

    result = []
    for i in sequences:
        result.append(kmerHits(minKmerLen, i))
    return result


def groupKmers(hitDicts, alphabet, similarity=0.7):
    kmersIter = kmersSetTuple(hitDicts)

    rows = len(alphabet)
    cols = len(kmersIter[0])
    zeros = np.zeros((rows, cols))
    result = {}
    for i in kmersIter:
        result.update({i: zeros})

    for _dict in hitDicts:
        for i in tuple(_dict.keys()):
            for j in kmersIter:
                if pairKmers(i, j, alphabet, similarity):
                    result[j] += kmerToFreq(i, alphabet) * _dict[i]
    return result


# testStr = getTestDNAStr(10000)
# testStrs = [getTestDNAStr(10) for i in range(2)]
# testDicts = seqsHits(5, testStrs)
# print(groupKmers(testDicts, alphabet=testDNA))

"""

reverse complement?

"""

# prueba = kmerHits(5, testStr)
# print(prueba)
# print("_" * 66)
# print("_" * 66)
# print("_" * 66)
# print(groupKmers(prueba, testDNA))


testSTR1 = "TTTTTACGTATTTTT"
# x = positionFreqNoise(5, testSTR1, testDNA, relative=True)
y = kmerHits(5, testSTR1, testDNA)
# print(x)
print(y)
print()
