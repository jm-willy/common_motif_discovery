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


def freqToSeq(freqMatrix, alphabet):
    """
    Convert a frequency matrix to a sequence string with ambiguity notation.

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


def randomDNA(length, GC=0.44):
    gc = GC / 2
    at = (1 - GC) / 2
    p = [at, gc, gc, at]
    return "".join(np.random.choice(testDNA, size=length, p=p))


def insertMotif(index, motif, seq):
    return seq[:index] + motif + seq[index:]


def dnaWithMotif(motifs, motifCount, seqLen, seqCount):
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


def hitUp(kmerStr, hitsDict):
    try:
        hitsDict[kmerStr] += 1
    except KeyError:
        hitsDict.update({kmerStr: 1})
    return hitsDict


def kmersSetTuple(hitDicts):
    kmersIter = set()
    for _dict in hitDicts:
        for i in tuple(_dict.keys()):
            kmersIter.add(i)
    return tuple(kmersIter)


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
