import unittest

from human_codon_usage_data import human_codon_usage_dict

# prueba = kmerHits(5, testStr)

# x = kmerToFreq("ACGTA", testDNA)
# y = kmerToFreq("ACGTA"[::-1], testDNA)
# x = kmerToFreq("ACGTA", testDNA)
# y = kmerToFreq("ACGTA"[::-1], testDNA)


# print(pairKmers("ACGTA", "GCGTT", alphabet=testDNA))


class TestDict(unittest.TestCase):
    def test_amino_acid_number(self):
        counter = 0
        for i in human_codon_usage_dict:
            for j in human_codon_usage_dict[i]:
                if j == "name":
                    continue
                counter += 1
        self.assertEqual(counter, 61)
        return

    def test_amino_acid_usage_sums(self):
        for i in human_codon_usage_dict:
            usage_sum = 0
            for j in human_codon_usage_dict[i]:
                if j == "name":
                    continue
                usage_sum += human_codon_usage_dict[i][j]
            # self.assertEqual(round(usage_sum, 2), 1, msg='amino acid {}'.format(i)) # does not allow delta
            # floating-point rounding error
            # self.assertAlmostEqual(usage_sum, 1, places=None, msg='amino acid {}'.format(i), delta=0.01)
            # AssertionError: 1.01 != 1 within 0.01 delta (0.010000000000000009 difference)
            # works with the raised delta
            # self.assertAlmostEqual(usage_sum, 1, places=None, msg='amino acid {}'.format(i), delta=0.010000000000000009)
            # solution
            self.assertLessEqual(usage_sum, 1.01, msg="amino acid {}".format(i))
            self.assertGreaterEqual(usage_sum, 0.99, msg="amino acid {}".format(i))
            # which is equal to
            self.assertLessEqual(usage_sum, 1 + 0.01, msg="amino acid {}".format(i))
            self.assertGreaterEqual(usage_sum, 1 - 0.01, msg="amino acid {}".format(i))
        return

    # def test_usage_list_same_length(self):
    #    for i in human_codon_usage_dict:
    #       aa_usage_lists = list(human_codon_usage_dict[i].values())[1:]
    #       list_length = len(aa_usage_lists[0])
    #       for j in aa_usage_lists[1:]:
    #          self.assertEqual(len(j), list_length, msg='amino acid {}'.format(i))
    #    return

    # def test_usage_list_usage_sums(self):
    #    for i in human_codon_usage_dict:
    #       aa_usage_lists = list(human_codon_usage_dict[i].values())[1:]
    #       for j in range(len(aa_usage_lists[0])):
    #          usage_sum = 0
    #          for k in aa_usage_lists:
    #             usage_sum += k[j]
    #          self.assertEqual(round(usage_sum, 2), 1, msg='amino acid {}'.format(i))
    #    return


# if __name__ == "__main__":
#     unittest.main()
