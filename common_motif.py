from typing import Any, Callable, Dict, List, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from utils import benchmark, dnaWithMotif, freqToSeq, getAll, hitUp, isProtein, seqToFreq, testDNA


def kmerHits(kmerLen: int, sequence: str) -> Dict[str, int]:
    """
    kmer hits for a of sequence

    Returns:
    dict: { 'kmer' : count } for each kmer of kmerLen in the input seq
    """
    result: Dict[str, int] = {}
    _len = 1 + len(sequence) - kmerLen
    for i in range(_len):
        slider = sequence[i : i + kmerLen]
        hitUp(slider, result)
    return result


def pairAminoAcidKmers(seq1: str, seq2: str, similarity: float = 0.6) -> bool:
    """
    similarity of seq1 and seq2

    Params:
    seq1: str
    seq2: str
    similarity: fraction of letters shared relative to length

    Returns:
    bool: whether seq1 and seq2 pair
    """
    _len = len(seq1)
    hits = 0
    for i in (seq1, seq1[::-1]):
        for j in range(_len):
            if i[j] == seq2[j]:
                hits += 1
        similarity = (_len * similarity) - 10**-6
        if hits >= similarity:
            return True
    return False


def pairNucleicKmers(seq1: str, seq2: str, similarity: float = 0.6) -> bool:
    """
    similarity of seq1 and seq2

    Params:
    seq1: str
    seq2: str
    similarity: fraction of letters shared relative to length

    Returns:
    bool: whether seq1 and seq2 pair
    """
    _len = len(seq1)
    hits = 0
    for i in tuple(getAll(seq1).values()):
        for j in range(_len):
            if i[j] == seq2[j]:
                hits += 1
        if hits >= similarity:
            return True
    return False


def getPairKmersFunc(sequences: str) -> Callable[[str, str, float], bool]:
    pairKmers = None
    isProteinBool = isProtein(sequences[0])
    if isProteinBool:
        pairKmers = pairAminoAcidKmers
    elif not isProteinBool:
        pairKmers = pairNucleicKmers
    elif isProteinBool is None:
        raise ValueError("isProteinBool must be True or False")
    else:
        raise ValueError("isProteinBool must be True or False")
    return pairKmers


def commonKmers(
    kmerLen: int,
    sequences: List[str],
    similarity: float = 0.6,
    occurrence: float = 0.6,
) -> Dict[str, int]:
    """
    common kmers across input sequences

    Params:
    similarity: fraction of letters shared to be regarded as same kmer
    occurrence: fraction of input sequences in which similar kmers should be found
    alphabet: list[str]

    Returns:
    dict: { kmer : self count } identical, not similar, count
    """

    occurrence -= 10**-6

    hitsList: List[Dict[str, int]] = []
    kmersLists: list[tuple[str, ...]] = []
    seqCount = 0
    unique: Set[str] = set()
    for seq in sequences:
        hits = kmerHits(kmerLen, seq)
        hitsList.append(hits)
        _kmers: Tuple[str, ...] = tuple(hits.keys())
        kmersLists.append(_kmers)
        unique.update(_kmers)
        seqCount += 1

    common: Dict[str, int] = {}
    for kmer in unique:
        common.update({kmer: 0})

    pairKmers = getPairKmersFunc(sequences)  # type: ignore

    iCount = 0
    for iList in kmersLists:
        for iKmer in iList:
            hits = 0
            jCount = 0
            for jList in kmersLists:
                if iCount == jCount:
                    jCount += 1
                    continue
                for jKmer in jList:
                    if pairKmers(iKmer, jKmer, similarity):
                        hits += 1
                jCount += 1
            if occurrence <= (hits / seqCount):
                common[iKmer] += hitsList[iCount][iKmer]
        iCount += 1

    toDel: List[str] = []
    for _key in common:
        if common[_key] == 0:
            toDel.append(_key)
    for kmer in toDel:
        del common[kmer]
    return common


def groupKmers(
    kmerLen: int,
    sequences: List[str],
    similarity: float = 0.7,
    occurrence: float = 0.6,
) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
    """
    kmers grouped by similarity

     Params:
     similarity: fraction of letters shared to group together

     Returns:
     { kmer : similar kmers } , { kmer : count }
    """
    common = commonKmers(kmerLen, sequences, similarity, occurrence)
    unique = tuple(common.keys())
    result: Dict[str, Set[str]] = {}
    for kmer in unique:
        result.update({kmer: set()})

    pairKmers = getPairKmersFunc(sequences)  # type: ignore

    for i in unique:
        for j in unique:
            if pairKmers(i, j, similarity):
                result[i].add(j)

    toDel: List[str] = []
    for _key in result:
        if len(result[_key]) == 0:
            toDel.append(_key)
    for kmer in toDel:
        del result[kmer]
    return result, common


def reduceGroup(
    kmerLen: int,
    sequences: List[str],
    alphabet: List[str],
    similarity: float = 0.7,
    occurrence: float = 0.6,
) -> Dict[str, NDArray[np.float64]]:
    """
    positional frequency for each kmer group

    Returns:
    dict: { kmer : nd.array }
    """
    groups, kmerCount = groupKmers(kmerLen, sequences, similarity, occurrence)
    result: Dict[str, NDArray[np.float64]] = {}
    unique = tuple(groups.keys())

    for _key in unique:
        freq = np.zeros((len(alphabet), kmerLen))
        for _similar in tuple(groups[_key]):
            freq += kmerCount[_similar] * seqToFreq(_similar, alphabet)

        freq += kmerCount[_key] * seqToFreq(_key, alphabet)
        result.update({_key: freq})
    return result


def letterFreqs(kmerLen: int, sequence: str, alphabet: List[str]) -> NDArray[np.floating[Any]]:
    """
    letter frequency for seq

    Returns:
    numpy.ndarray: positional frequency matrix
    """
    sliderLen = 1
    result: NDArray[np.floating[Any]] = np.zeros((len(alphabet), sliderLen))
    _len = 1 + len(sequence) - sliderLen
    for i in range(_len):
        slider = sequence[i : i + sliderLen]
        result += seqToFreq(slider, alphabet)
    result = result / np.sum(result)
    result = np.concat([result for i in range(kmerLen)], axis=-1)  # type: ignore
    return result


def kmerConvolution(x: str, y: str, alphabet: List[str]):
    """
    no-gap align 2 same length kmers seqs through convolution
    x is expanded, while y is slided over x like a convolutional kernel

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
    aligns: List[NDArray[np.floating[Any]]] = []
    for i in range(_len):
        slider = xFreq[i : i + kmerLen]
        freq = (slider + yFreq) + (slider * yFreq)
        aligns.append(freq)

    sums: List[np.floating[Any]] = []
    for i in aligns:
        sums.append(np.sum(i))

    _max: np.floating[Any] = np.max(sums)
    count = 0
    result = np.array([])
    for i in sums:
        if i >= _max:
            result = aligns[count].T
        count += 1
    return result


def kmerConv(x: str, y: str, alphabet: List[str]):
    """
    get the best positional freq matrix for x and y
    seqs strs among both forward and reversed input
    """
    a = kmerConvolution(x, y, alphabet)
    b = kmerConvolution(x[::-1], y, alphabet)
    return a if np.sum(a) >= np.sum(b) else b.T[::-1].T


def growKmer():
    "grow kmers by similarity comparison in a func similar to kmerHit()"
    "replace threshold for similarity, start from highest similarity go low"
    "test more different motifs, however it should spot ACA and AGA"
    "iterate while"
    "groupReduce, get best matrix instead?"
    "reduce until similarity is 1 to get the pure hits, should get 2 motifs aga and aca"
    "apply group reduce only not kmer conv"
    "reverse input sequences?"
    # _len = 1 + len(sequence) - kmerLen
    #     for i in range(_len):
    #         slider = sequence[i : i + kmerLen]
    return


@benchmark
def extractFinal():
    print("\n" * 2)
    s0 = dnaWithMotif(["GCTGG", "TACGT"], motifCount=2, seqLen=10, seqCount=7)
    # for i in s0:
    #     print(i)
    # x = commonGroups(5, s0, similarity=4 / 5, occurrence=0.6)
    # print("TACGT in", "TACGT" in x[0].keys(), "TACGT" in x[1].keys())
    # print("GCTGG in", "GCTGG" in x[0].keys(), "GCTGG" in x[1].keys())
    # print()
    x = reduceGroup(5, s0, testDNA, similarity=4 / 5, occurrence=0.6)
    for i in x:
        print(i, freqToSeq(x[i], testDNA))
        print(x[i])
        print()
    return


extractFinal()
