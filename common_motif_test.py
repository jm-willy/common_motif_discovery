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

    def test_pairAminoAcidKmers(self):
        x = pairNucleicKmers("DEVD", "DEVD", similarity=1)
        self.assertTrue(x)
        x = pairNucleicKmers("YGQW", "YGGF", similarity=1)
        self.assertTrue(not x)
        return

    def test_pairNucleicKmers(self):
        x = pairNucleicKmers("ACGTA", "ACGTA", similarity=1)
        self.assertTrue(x)
        x = pairNucleicKmers("ACGTA", "ACGTT", similarity=1)
        self.assertTrue(not x)
        x = pairNucleicKmers("AC", "AT", similarity=0.5)
        self.assertTrue(x)
        return

    def test_kmerHits(self):
        x = kmerHits(5, "TTTTTACGTATTTTT")
        self.assertTrue(x["ACGTA"] == 1)
        x = kmerHits(7, "TTTTTACGTATTTTT")
        self.assertTrue(x["TACGTAT"] == 1)
        x = kmerHits(5, "TTTTTACGTATTTTT")
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

    def test_commonKmers(self):
        s1 = "TTTTTGGGG"
        s2 = s1.replace("T", "A")
        x = commonKmers(5, (s1, s2), similarity=4 / 5, occurrence=0.5)
        self.assertTrue(x[0] == "TGGGG")
        self.assertTrue(x[1] == "AGGGG")
        return

    def test_getComplementary(self):
        x = getComplementary("ACTGA")
        self.assertTrue(x == "TGACT")
        return

    def test_getAll(self):
        x = getAll("ACTGA")
        self.assertTrue(x["input"] == "ACTGA")
        self.assertTrue(x["reverse"] == "AGTCA")
        self.assertTrue(x["complementary"] == "TGACT")
        self.assertTrue(x["reverse_complementary"] == "TCAGT")
        return


if __name__ == "__main__":
    unittest.main()
