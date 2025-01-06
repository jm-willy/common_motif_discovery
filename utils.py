from collections import OrderedDict

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


def normalize_to_0_1(matrix):
    return (np.array(matrix) - np.min(matrix)) / (np.max(matrix) - np.min(matrix))


def getTestDNAStr(length, GC=0.44):
    gc = GC / 2
    at = (1 - GC) / 2
    p = [at, gc, gc, at]
    return "".join(np.random.choice(testDNA, size=length, p=p))


def orderHits(hitsDict):
    result = OrderedDict(sorted(hitsDict.items(), key=lambda x: x[1], reverse=True))
    return result


def orderGroups(groupsDict):
    result = OrderedDict(sorted(groupsDict.items(), key=lambda x: np.sum(x[1]), reverse=True))
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


def freqToSeq(freqMatrix, alphabet, skipDraw=False):
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


# def filterHits(hitDicts):
#     kmersIter = kmersSetTuple(hitDicts)

#     for _dict in hitDicts:
#         kmersKeys = tuple(_dict.keys())
#         for i in kmersKeys:
#             if i not in kmersIter:
#                 del _dict[i]
#     return hitDicts
