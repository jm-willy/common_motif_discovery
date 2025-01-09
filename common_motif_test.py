import unittest

from common_motif import *
from utils import *


class TestDict(unittest.TestCase):
    def test_seqToFreq(self):
        x = seqToFreq("ACGTA", testDNA)
        self.assertEqual(x[0][0], 1)
        self.assertEqual(x[1][1], 1)
        self.assertEqual(x[3][3], 1)
        return

    def test_freqToSeq(self):
        x = seqToFreq("ACGTA", testDNA)
        x = freqToSeq(x, testDNA)
        self.assertTrue(x == "ACGTA")
        return

    def test_pairKmers(self):
        x = pairKmers("ACGTA", "ACGTA", alphabet=testDNA, similarity=1)
        self.assertTrue(x)
        x = pairKmers("ACGTA", "ACGTT", alphabet=testDNA, similarity=1)
        self.assertTrue(not x)
        x = pairKmers("AC", "AT", alphabet=testDNA, similarity=0.5)
        self.assertTrue(x)
        return

    def test_kmerHits(self):
        x = kmerHits(5, "TTTTTACGTATTTTT", alphabet=testDNA)
        self.assertTrue(x["ACGTA"] == 1)
        x = kmerHits(7, "TTTTTACGTATTTTT", alphabet=testDNA)
        self.assertTrue(x["TACGTAT"] == 1)
        x = kmerHits(5, "TTTTTACGTATTTTT", testDNA)
        self.assertTrue(x["TTTTT"] == 2)
        return

    def test_insertMotif(self):
        x = insertMotif(0, "A", "TG")
        self.assertTrue(x == "ATG")
        x = insertMotif(1, "A", "TG")
        self.assertTrue(x == "TAG")
        x = insertMotif(2, "A", "TG")
        self.assertTrue(x == "TGA")
        return

    def test_dnaWithMotif(self):
        x = dnaWithMotif(["ACGTA"], motifCount=1, seqLen=10, seqCount=3)
        x = ["ACGTA" in i for i in x]
        self.assertTrue(all(x))
        return


if __name__ == "__main__":
    unittest.main()
