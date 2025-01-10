import numpy as np
from utils import benchmark, dnaWithMotif, freqToSeq, hitUp, kmersSetTuple, testDNA


def seqToFreq(kmer, alphabet):
    if type(kmer) is not str:
        raise ValueError("kmer must be a str")
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


def kmerHits(kmerLen, sequence):
    """
    kmer hits for a list/tuple of sequences

    Returns:
    dict: { 'kmers' : count } for each kmer of kmerLen in the input seq
    """

    result = {}
    _len = 1 + len(sequence) - kmerLen
    for i in range(_len):
        slider = sequence[i : i + kmerLen]
        hitUp(slider, result)
    return result


def pairKmers(seq1, seq2, alphabet, similarity=0.7):
    """
    Params:
    seq1: str
    seq2: str
    similarity: fraction of letters shared to group together

    Returns:
    bool: whether seq1 and seq2 pair
    """
    _len = 0
    similarity = len(seq1) * similarity
    x = seqToFreq(seq1, alphabet)
    y = seqToFreq(seq2, alphabet)
    score = np.sum(x * y)
    # if score < similarity:
    #     x = seqToFreq(seq1[::-1], alphabet)
    #     score = np.sum(x * y)
    return score >= (similarity - 10**-6)


def groupKmers(kmerLen, sequence, alphabet, similarity=0.7):
    """
    Params:
    similarity: fraction of letters shared to group together
    threshold: kmer fold enrichment as a fraction of the most fold-enriched  kmer (1.0) per seq

    Returns:
    dict , int: { str : { str : int } } , total hits
    """
    hits = kmerHits(kmerLen, sequence)
    hits = hits, hits
    unique = kmersSetTuple(hits)

    result = {}
    for i in unique:
        result.update({i: {}})

    # count = 0
    for _dict in hits:
        for i in tuple(_dict.keys()):
            for j in unique:
                if j == i:
                    continue
                if pairKmers(i, j, alphabet, similarity):
                    result[j].update({i: _dict[i]})
                    # count += 1

    toDel = []
    for _key in result:
        if len(result[_key]) == 0:
            toDel.append(_key)
    for i in toDel:
        del result[i]
    return result


def groupCommon(kmerLen, sequence, alphabet, similarity=0.7):
    """
    Params:
    similarity: fraction of letters shared to group together
    threshold: kmer fold enrichment as a fraction of the most fold-enriched  kmer (1.0) per seq

    Returns:
    dict , int: { str : { str : int } } , total hits
    """

    common = commonKmers(kmerLen, sequence, alphabet, similarity)
    unique = set()
    for _list in common:
        unique.union(set(_list))
    unique = tuple(unique)

    result = {}
    for kmer in unique:
        result.update({kmer: []})

    iCount = 0
    for iList in common:
        jCount = 0
        for jList in common:
            if iCount == jCount:
                jCount += 1
                continue
            else:
                iKmer = iList[iCount]
                jKmer = jList[jCount]
                if pairKmers(iKmer, jKmer, alphabet, similarity):
                    result[iKmer].append(jKmer)
                jCount += 1
        iCount += 1

    toDel = []
    for _key in result:
        if len(result[_key]) == 0:
            toDel.append(_key)
    for kmer in toDel:
        del result[kmer]
    return result


def commonKmers(kmerLen, sequences, alphabet, similarity=0.6):
    """
    kmer common to sequences
    """
    if type(sequences) is not list and type(sequences) is not tuple:
        raise ValueError("sequences type must be list[str]")

    hitsList = []
    kmersLists = []
    for seq in sequences:
        hits = kmerHits(kmerLen, seq)
        hitsList.append(hits)
        kmersLists.append([i for i in tuple(hits.keys())])

    common = []
    iCount = 0
    for i_list in kmersLists:
        jCount = 0
        for j_list in kmersLists:
            if iCount == jCount:
                jCount += 1
                continue
            else:
                for i_kmer in i_list:
                    for j_kmer in j_list:
                        if pairKmers(i_kmer, j_kmer, alphabet, similarity):
                            common += [i_kmer for i in range(hitsList[iCount][i_kmer])]
                jCount += 1
        iCount += 1

    # count = 0
    # score = count / len(sequences)

    # result = {}
    return common


def letterFreqs(kmerLen, sequence, alphabet):
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
        raise ValueError("kmerConvolution() different x and y lengths")
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


def reduceGroup(freqs, alphabet):
    """replace with multiple seq align"""
    if len(freqs) == 0:
        raise ValueError("reduce() given freqs is empty")
    if type(freqs[0]) is str:
        _list = []
        for i in freqs:
            _list.append(seqToFreq(i, alphabet))
        freqs = _list

    _mul = np.ones(freqs[0].shape)
    ## very large (and small) numbers =  NaN error
    for i in freqs:
        _mul *= i
    _sum = np.zeros(freqs[0].shape)
    for i in freqs:
        _sum += i
    result = _mul + _sum
    return result


# def motifGroup(kmerLen, sequences, alphabet, threshold=0.9, similarity=0.7):
#     """
#     Params:
#     similarity: fraction of letters shared to group together
#     threshold: kmer fold enrichment as a fraction of the most fold-enriched  kmer (1.0) per seq

#     Returns:
#     dict: { str : ndarray }
#     """
#     g, c = groupKmers(kmerLen, sequences, alphabet, similarity=0.7)
#     result = g
#     for key_i in tuple(g.keys()):
#         convs = []
#         for key_j in tuple(g[key_i].keys()):
#             # x = kmerConv(_key, i, alphabet) * g[_key][i]
#             # x = freqToSeq(x, alphabet)
#             convs.append(kmerConv(key_i, key_j, alphabet) * g[key_i][key_j])
#         multi = reduceGroup(convs, alphabet)
#         # result[key_i] = freqToSeq(multi, alphabet)
#         result[key_i] = multi
#     return result


# def motifForLen(kmerLen, sequences, alphabet, threshold=0.9, similarity=0.7):
#     """
#     Params:
#     similarity: fraction of letters shared to group together
#     threshold: kmer fold enrichment as a fraction of the most fold-enriched  kmer (1.0) per seq

#     Returns:
#     list: [kmers]
#     """

#     mg, c = motifGroup(kmerLen, sequences, alphabet, threshold=0.9, similarity=0.7)
#     # groupKmers()
#     x = reduceGroup(tuple(mg.values()), alphabet)
#     return freqToSeq(x, alphabet)


def growKmer():
    "grow kmers by similarity comparison in a func similar to kmerHit()"
    "replace threshold for similarity, start from highest similarity go low"
    "test more different motifs, however it should spot ACA and AGA"
    "iterate while"
    "groupReduce, get best matrix instead?"
    "reduce until similarity is 1 to get the pure hits, should get 2 motifs aga and aca"
    "apply group reduce only not kmer conv"
    # _len = 1 + len(sequence) - kmerLen
    #     for i in range(_len):
    #         slider = sequence[i : i + kmerLen]
    return


@benchmark
def extractFinal():
    print("\n" * 2)
    s0 = dnaWithMotif(["GCTGG", "TACGT"], motifCount=2, seqLen=1000, seqCount=7)
    # s0 = dnaWithMotif(["TTTT"], motifCount=2, seqLen=20, seqCount=6)
    x = commonKmers(5, s0, testDNA, similarity=1)
    # print(s0)
    # s1 = s0[0]
    # s2 = s0[1]
    _list = []
    for i in x:
        for j in x:
            q = kmerConvolution(i, j, testDNA)
            _list.append(q)
            # q = freqToSeq(q, testDNA)
            # print(q)
    # print(_list)
    x = reduceGroup(_list, testDNA)
    print(x / np.max(x))
    x = freqToSeq(x, testDNA)
    print(x)
    # x = matches(3, (x, x), testDNA, similarity=0.6)
    # print(x)
    # _list = []
    # for i in x:
    #     for j in x:
    #         q = kmerConvolution(i, j, testDNA)
    #         _list.append(q)
    # x = groupReduce(_list, testDNA)
    # x = freqToSeq(x, testDNA)
    # print(x)
    return


extractFinal()
