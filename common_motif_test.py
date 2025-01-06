import unittest

from common_motif import *
from utils import *


class TestDict(unittest.TestCase):
    def test_kmerToFreq(self):
        x = kmerToFreq("ACGTA", testDNA)
        self.assertEqual(x[0][0], 1)
        self.assertEqual(x[1][1], 1)
        self.assertEqual(x[3][3], 1)
        return

    def test_pairKmers(self):
        x = pairKmers("ACGTA", "ACGTA", alphabet=testDNA, similarity=1)
        self.assertTrue(x)
        x = pairKmers("ACGTA", "ACGTT", alphabet=testDNA, similarity=1)
        self.assertTrue(not x)
        x = pairKmers("AC", "AT", alphabet=testDNA, similarity=0.5)
        self.assertTrue(x)
        "TTTTTACGTATTTTT"
        return

    def test_kmerHits(self):
        x = kmerHits(5, "TTTTTACGTATTTTT", alphabet=testDNA)
        self.assertTrue(x["ACGTA"] == 1)
        x = kmerHits(7, "TTTTTACGTATTTTT", alphabet=testDNA)
        self.assertTrue(x["TACGTAT"] == 1)
        return


if __name__ == "__main__":
    unittest.main()
