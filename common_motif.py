import numpy as np
from utils import freqToSeq, hitUp, kmersSetTuple, randomDNA, testDNA


def seqToFreq(kmer, alphabet):
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


def freqNoise(kmerLen, sequence, alphabet):
    """
    letter frequency for seq

    Returns:
    numpy.ndarray: positional frequency matrix
    """

    sliderLen = 1
    result = np.zeros((len(alphabet), sliderLen))
    _len = 1 + len(sequence) - sliderLen
    for i in range(_len):
        slider = sequence[i : i + sliderLen]
        result += seqToFreq(slider, alphabet)
    result = result / np.sum(result)
    result = np.concat([result for i in range(kmerLen)], axis=-1)
    return result


def kmerEnrichment(kmerLen, sequence, alphabet):
    """
    enrichment = enrichment / np.max(enrichment)
    compared to background

    Returns:
    numpy.ndarray: of len(sequence)
    """
    e = 10**-6
    enrichment = []
    _len = 1 + len(sequence) - kmerLen
    noise = freqNoise(kmerLen, sequence, alphabet)
    for i in range(_len):
        slider = sequence[i : i + kmerLen]
        fold = seqToFreq(slider, alphabet) / (noise - e)
        enrichment.append(np.sum(fold))
    enrichment = enrichment / np.max(enrichment)
    return enrichment


def kmerHits(kmerLen, sequence, alphabet, threshold=0.51):
    """
    kmer hits for a single sequence above the enrichment threshold
    compared to background positional frequency matrix

    compute kmer enrichment assuming flat unifor distribution for length

    Params:
    threshold: fold enrichment as a fraction of the most fold-enriched kmer (1.0)

    Returns:
    dict: { 'kmers' : count } for each kmer of kmerLen in the input seq
    """
    _len = 1 + len(sequence) - kmerLen
    enrichment = kmerEnrichment(kmerLen, sequence, alphabet)
    e = 10**-6
    result = {}
    for i in range(_len):
        slider = sequence[i : i + kmerLen]
        if enrichment[i] >= (threshold - e):
            hitUp(slider, result)
    return result


def seqsHits(kmerLen, sequences, alphabet, threshold=0.9):
    """
    kmer hits per sequence for several sequences

    Params:
    sequences: list/tuple of strings
    threshold: kmer fold enrichment as a fraction of the most fold-enriched  kmer (1.0) per seq

    Returns:
    list: list of dict { 'kmers' : count } per input sequence
    """
    result = []
    for i in sequences:
        result.append(kmerHits(kmerLen, i, alphabet, threshold))
    return result


def kmerConvolution(x, y, alphabet):
    """
    no-gap align 2 same length kmers seqs through convolution
    x is expanded, while y is slided over x like a convolutional kernel

    Params:
    op: sum or dot the freq matrices

    Returns:
    np.ndarray: positional freq matrix
    """
    if len(x) != len(y):
        raise ValueError("kmerPairAlign different x and y lengths")
    xFreq = seqToFreq(x, alphabet)
    zeros = np.zeros(xFreq.shape)
    kmerLen = xFreq.shape[-1]
    xFreq = np.concat((zeros, xFreq, zeros), axis=-1).T
    yFreq = seqToFreq(y, alphabet).T
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


def kmerConv(x, y, alphabet):
    """
    get the best positional freq matrix for x and y
    seqs strs among both forward and reversed input
    """
    a = kmerConvolution(x, y, alphabet)
    b = kmerConvolution(x[::-1], y, alphabet)
    return a if np.sum(a) >= np.sum(b) else b.T[::-1].T


def groupReduce(freqs, alphabet):
    _mul = np.ones(freqs[0].shape)
    for i in freqs:
        _mul *= i
    _sum = np.zeros(freqs[0].shape)
    for i in freqs:
        _sum += i
    result = _mul + _sum
    return result


def pairKmers(seq1, seq2, alphabet, similarity=0.7):
    """
    Params:
    similarity: fraction of letters shared to group together

    Returns:
    bool: whether seq1 and seq2 pair
    """
    similarity = len(seq1) * similarity
    x = seqToFreq(seq1, alphabet)
    y = seqToFreq(seq2, alphabet)
    score = np.sum(x * y)
    if score < similarity:
        x = seqToFreq(seq1[::-1], alphabet)
        score = np.sum(x * y)
    return score >= similarity


def groupKmers(kmerLen, sequences, alphabet, threshold=0.9, similarity=0.7):
    """
    Params:
    similarity: fraction of letters shared to group together
    threshold: kmer fold enrichment as a fraction of the most fold-enriched  kmer (1.0) per seq

    Returns:
    dict: { str : { str : int } }
    """
    seqsHitsIter = seqsHits(kmerLen, sequences, alphabet, threshold=threshold)
    kmersIter = kmersSetTuple(seqsHitsIter)
    result = {}
    for i in kmersIter:
        result.update({i: {}})

    for _dict in seqsHitsIter:
        for i in tuple(_dict.keys()):
            for j in kmersIter:
                if j == i:
                    continue
                if pairKmers(i, j, alphabet, similarity):
                    result[j].update({i: _dict[i]})

    toDel = []
    for _key in result:
        if len(result[_key]) == 0:
            toDel.append(_key)
    for i in toDel:
        del result[i]
    return result


def motifGroup(kmerLen, sequences, alphabet, threshold=0.9, similarity=0.7):
    """
    Params:
    similarity: fraction of letters shared to group together
    threshold: kmer fold enrichment as a fraction of the most fold-enriched  kmer (1.0) per seq

    Returns:
    dict: { str : ndarray }
    """
    g = groupKmers(kmerLen, sequences, alphabet, threshold=0.9, similarity=0.7)
    result = g
    for key_i in tuple(g.keys()):
        convs = []
        kmers = tuple(g[key_i].keys())
        for key_j in kmers:
            # x = kmerConv(_key, i, alphabet) * g[_key][i]
            # x = freqToSeq(x, alphabet)
            convs.append(kmerConv(key_i, key_j, alphabet) * g[key_i][key_j])
        multi = groupReduce(convs, alphabet)
        # result[key_i] = freqToSeq(multi, alphabet)
        result[key_i] = multi
    return result


def motifForLen(kmerLen, sequences, alphabet, threshold=0.9, similarity=0.7):
    """
    Params:
    similarity: fraction of letters shared to group together
    threshold: kmer fold enrichment as a fraction of the most fold-enriched  kmer (1.0) per seq

    Returns:
    list: [kmers]
    """

    "test more different motifs, however it should spot ACA and AGA"
    "iterate while"
    "groupReduce, get best matrix instead?"
    "reduce until similarity is 1 to get the pure hits, should get 2 motifs aga and aca"
    "apply group reduce only not kmer conv"
    mg = motifGroup(kmerLen, sequences, alphabet, threshold=0.9, similarity=0.7)
    groupKmers()
    x = groupReduce(tuple(mg.values()), alphabet)
    return freqToSeq(x, alphabet)


def growKmer():
    "grow kmers by similarity comparison in a func similar to kmerHit()"
    "replace threshold for similarity"
    return


# def growKmer():
#     "grow kmers by similarity comparison in a func similar to kmerHit()"
#     "replace threshold for similarity"

#     _len = 1 + len(sequence) - kmerLen
#     enrichment = letterEnrichment(kmerLen, sequence, alphabet)
#     e = 10**-6
#     result = {}
#     for i in range(_len):
#         slider = sequence[i : i + kmerLen]
#         if enrichment[i] >= (threshold - e):
#             hitUp(slider, result)
#     return result


# s1 = "TTTTTACATTTTAGATTTTTT"
# s2 = s1.replace("T", "G")
# L = (s1, s2)
# # # y = kmerHits(4, testSTR1, testDNA, threshold=1)
# # # print(y)
# # # x = getSeqsHits(5, L, alphabet=testDNA)

# x = groupKmers(5, L, alphabet=testDNA, threshold=0.9)
# print(x)
# print("\n\n")
# x = motifGroup(5, L, alphabet=testDNA, threshold=0.9)
# print(x)
# print("\n\n")
# x = motifForLen(5, L, alphabet=testDNA, threshold=0.9)
# print(x)

# # x = groupKmers(5, L, alphabet=testDNA, threshold=0.9, similarity=0.7)
# # print(x)
# # print()
# # print(x.keys())
# # for i in tuple(x.values()):
# #     print(i)
# #     print(freqToSeq(i, testDNA))
# #     print()

# # s1 = "TTTACATTTAGA"
# # s2 = s1.replace("T", "G")
# # rs = kmerConv(s1, s2[::-1], testDNA)
# # print(rs)
# # print(np.sum(rs))
# # print(freqToSeq(rs, testDNA))
