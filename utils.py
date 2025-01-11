import random as r
import time

import numpy as np

alphabetDNA = ["-", "A", "C", "G", "T"]
alphabetRNA = ["-", "A", "C", "G", "T", "U"]
alphabetProtein = [
    "-",
    "A",
    "R",
    "N",
    "D",
    "C",
    "E",
    "Q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]

testDNA = alphabetDNA[1:]
testRNA = alphabetRNA[1:]
testProtein = alphabetProtein[1:]


def benchmark(func):
    loops = 1

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = None
        for i in range(loops):
            result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.6f} seconds.")
        return result

    return wrapper


def seqToFreq(sequence, alphabet):
    """
    convert sequence string to frequency matrix

    Params:
    kmer: str

    Returns:
    numpy.ndarray: positional frequency Matrix
    """
    if type(sequence) is not str:
        raise ValueError("kmer must be a str")

    matrix = []
    for i in alphabet:
        row = []
        for j in sequence:
            if i == j:
                row.append(1)
            else:
                row.append(0)
        matrix.append(row)
    return np.array(matrix)


def freqToSeq(freqMatrix, alphabet):
    """
    convert a frequency matrix to a sequence string with ambiguity notation

    Parameters:
    freqMatrix (np.ndarray): Matrix of symbol row frequencies per position
    alphabet (list): Ordered alphabet corresponding to matrix rows

    Returns:
    str: Sequence with ambiguity notation using square brackets
    """
    freqMatrix = np.array(freqMatrix)
    sequence = ""
    for col in range(freqMatrix.shape[-1]):
        freqs = freqMatrix[:, col]
        max_freq = np.max(freqs)
        max_indices = np.where(freqs == max_freq)[0]
        pos_alphabet = [alphabet[i] for i in max_indices]
        if len(pos_alphabet) == 1:
            sequence += pos_alphabet[0]
        else:
            sequence += f"[{''.join(pos_alphabet)}]"
    return sequence


def hitUp(kmerStr, hitsDict):
    try:
        hitsDict[kmerStr] += 1
    except KeyError:
        hitsDict.update({kmerStr: 1})
    return hitsDict


def randomDNA(length, GC=0.44):
    gc = GC / 2
    at = (1 - GC) / 2
    p = [at, gc, gc, at]
    return "".join(np.random.choice(testDNA, size=length, p=p))


def insertMotif(index, motif, seq):
    return seq[:index] + motif + seq[index:]


def dnaWithMotif(motifs, motifCount, seqLen, seqCount):
    """
    get random DNA sequences with randmly inserted motifs.
    For more than 1 motifs, motifs may randomly replace other motifs

    Returns: list[str]
    """
    if type(motifs) is not list and type(motifs) is not tuple:
        raise ValueError("motifs type must be list[str]")

    result = []
    for i in range(seqCount):
        seq = randomDNA(seqLen)
        for j in motifs:
            for k in range(motifCount):
                index = r.randint(0, seqLen)
                seq = insertMotif(index, j, seq)
        result.append(seq)
    return result


def getComplementaryDNA(sequence):
    result = []
    for i in sequence:
        if i == "A":
            result.append("T")
        elif i == "C":
            result.append("G")
        elif i == "G":
            result.append("C")
        elif i == "T":
            result.append("A")
    return "".join(result)
