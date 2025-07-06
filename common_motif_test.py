import unittest

import common_motif
import utils


class TestDict(unittest.TestCase):
    def test_seqToFreq(self):
        x = utils.seqToFreq("ACGTA", utils.testDNA)
        self.assertEqual(x[0][0], 1)
        self.assertEqual(x[1][1], 1)
        self.assertEqual(x[3][3], 1)
        return

    def test_freqToSeq(self):
        x = utils.seqToFreq("ACGTA", utils.testDNA)
        x = utils.freqToSeq(x, utils.testDNA)
        self.assertTrue(x == "ACGTA")
        return

    def test_pairAminoAcidKmers(self):
        x = common_motif.pairNucleicKmers("DEVD", "DEVD", similarity=1)
        self.assertTrue(x)
        x = common_motif.pairNucleicKmers("YGQW", "YGGF", similarity=1)
        self.assertTrue(not x)
        return

    def test_pairNucleicKmers(self):
        x = common_motif.pairNucleicKmers("ACGTA", "ACGTA", similarity=1)
        self.assertTrue(x)
        x = common_motif.pairNucleicKmers("ACGTA", "ACGTT", similarity=1)
        self.assertTrue(not x)
        x = common_motif.pairNucleicKmers("AC", "AT", similarity=0.5)
        self.assertTrue(x)
        return

    def test_kmerHits(self):
        x = common_motif.kmerHits(5, "TTTTTACGTATTTTT")
        self.assertTrue(x["ACGTA"] == 1)
        x = common_motif.kmerHits(7, "TTTTTACGTATTTTT")
        self.assertTrue(x["TACGTAT"] == 1)
        x = common_motif.kmerHits(5, "TTTTTACGTATTTTT")
        self.assertTrue(x["TTTTT"] == 2)
        return

    def test_insertMotif(self):
        x = utils.insertMotif(0, "A", "TG")
        self.assertTrue(x == "ATG")
        x = utils.insertMotif(1, "A", "TG")
        self.assertTrue(x == "TAG")
        x = utils.insertMotif(2, "A", "TG")
        self.assertTrue(x == "TGA")
        return

    def test_dnaWithMotif(self):
        x = utils.dnaWithMotif(["ACGTA"], motifCount=1, seqLen=10, seqCount=3)
        x = ["ACGTA" in i for i in x]
        self.assertTrue(all(x))
        return

    def test_commonKmers(self):
        s1 = "TTTTTGGGG"
        s2 = s1.replace("T", "A")
        x = common_motif.commonKmers(5, [s1, s2], similarity=4 / 5, occurrence=0.5)
        self.assertTrue(x["TGGGG"] == 1)
        self.assertTrue(x["AGGGG"] == 1)
        return

    def test_getComplementary(self):
        x = utils.getComplementary("ACTGA")
        self.assertTrue(x == "TGACT")
        return

    def test_getAll(self):
        x = utils.getAll("ACTGA")
        self.assertTrue(x["input"] == "ACTGA")
        self.assertTrue(x["reverse"] == "AGTCA")
        self.assertTrue(x["complementary"] == "TGACT")
        self.assertTrue(x["reverse_complementary"] == "TCAGT")
        return


if __name__ == "__main__":
    unittest.main()
