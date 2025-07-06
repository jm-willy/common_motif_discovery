import random as r
import time
from typing import Any, Dict, List

import numpy as np
from numpy.typing import NDArray

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


def benchmark(func):  # type: ignore
    loops = 1

    def wrapper(*args, **kwargs):  # type: ignore
        start_time = time.time()
        result = func(*args, **kwargs)  # type: ignore
        for i in range(loops):
            func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.6f} seconds.")  # type: ignore
        return result  # type: ignore

    return wrapper  # type: ignore


def seqToFreq(sequence: str, alphabet: List[str]) -> NDArray[np.float64]:
    """
    convert sequence string to frequency matrix

    Params:
    kmer: str

    Returns:
    numpy.ndarray: positional frequency Matrix
    """
    if type(sequence) is not str:
        raise ValueError("kmer must be a str")

    matrix: List[List[float]] = []
    for i in alphabet:
        row: List[float] = []
        for j in sequence:
            if i == j:
                row.append(1)
            else:
                row.append(0)
        matrix.append(row)
    return np.array(matrix)


def freqToSeq(frequencyMatrix: NDArray[np.floating[Any]], alphabet: List[str]) -> str:
    """
    convert a frequency matrix to a sequence string with ambiguity notation

    Parameters:
    freqMatrix (np.ndarray): Matrix of symbol row frequencies per position
    alphabet (list): Ordered alphabet corresponding to matrix rows

    Returns:
    str: Sequence with ambiguity notation using square brackets
    """
    freqMatrix: NDArray[np.floating[Any]] = np.array(frequencyMatrix)
    sequence = ""
    for col in range(freqMatrix.shape[-1]):
        freqs = freqMatrix[:, col]
        max_freq = np.max(freqs)
        max_indices = np.where(freqs == max_freq)[0]
        pos_alphabet: List[str] = [alphabet[i] for i in max_indices]
        if len(pos_alphabet) == 1:
            sequence += pos_alphabet[0]
        else:
            sequence += f"[{''.join(pos_alphabet)}]"
    return sequence


def hitUp(kmerStr: str, hitsDict: Dict[str, int]):
    try:
        hitsDict[kmerStr] += 1
    except KeyError:
        hitsDict.update({kmerStr: 1})
    return hitsDict


def randomDNA(length: int, GC: float = 0.44):
    gc = GC / 2
    at = (1 - GC) / 2
    p = [at, gc, gc, at]
    return "".join(np.random.choice(testDNA, size=length, p=p))


def insertMotif(index: int, motif: str, seq: str):
    return seq[:index] + motif + seq[index:]


def dnaWithMotif(motifs: List[str], motifCount: int, seqLen: int, seqCount: int) -> List[str]:
    """
    get random DNA sequences with randmly inserted motifs.
    For more than 1 motifs, motifs may randomly replace other motifs

    Returns: seqCount list[str] motifs randomly inserted
    """

    result: List[str] = []
    for i in range(seqCount):
        seq = randomDNA(seqLen)
        for j in motifs:
            for k in range(motifCount):
                index = r.randint(0, seqLen)
                seq = insertMotif(index, j, seq)
        result.append(seq)
    return result


def getComplementary(sequence: str) -> str:
    result: List[str] = []
    for i in sequence:
        if i == "A":
            result.append("T")
        elif i == "C":
            result.append("G")
        elif i == "G":
            result.append("C")
        elif i == "T":
            result.append("A")
        elif i == "U":
            result.append("A")
    return "".join(result)


def getAll(sequence: str) -> Dict[str, str]:
    complementary = getComplementary(sequence)
    return {
        "input": sequence,
        "reverse": sequence[::-1],
        "complementary": complementary,
        "reverse_complementary": complementary[::-1],
    }


def isProtein(sequence: str) -> bool:
    """
    Will fail to tell apart proteins whose
    only amino acids are "A", "C", "G", "T"
    """
    if "U" in sequence:
        return False

    unique = tuple(set(alphabetProtein) - set(alphabetDNA))
    for i in unique:
        if i in sequence:
            return True
    return False
