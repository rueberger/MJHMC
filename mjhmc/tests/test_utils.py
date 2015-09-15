import unittest
import numpy as np
from misc.utils import min_idx

n_seed = 1
list_length = 100

class TestMinIdx(unittest.TestCase):
    """test that min idx works
    """

    def setUp(self):
        np.random.seed(n_seed)

    def test_two_case(self):
        """
        tests the very important two list case
        """
        list_1 = np.random.randn(list_length)
        list_2 = np.random.randn(list_length)
        mi_1_test, mi_2_test = min_idx(
            [list_1.reshape(1, list_length),
             list_2.reshape(1, list_length)])

        mi_1_control = np.arange(list_length)[list_1 < list_2]
        mi_2_control = np.arange(list_length)[list_1 >= list_2]
        self.assertTrue((mi_1_test == mi_1_control).all(), "idx minimums do not match")
        self.assertTrue((mi_2_test == mi_2_control).all(), "idx minimums do not match")

    def test_three_case(self):
        """
        tests the three list case
        """
        list_1 = np.random.randn(list_length)
        list_2 = np.random.randn(list_length)
        list_3 = np.random.randn(list_length)
        mi_1_test, mi_2_test, mi_3_test = min_idx(
            [list_1.reshape(1, list_length),
             list_2.reshape(1, list_length),
             list_3.reshape(1, list_length)])

        mi_12_control = set(np.arange(list_length)[list_1 < list_2])
        mi_13_control = set(np.arange(list_length)[list_1 < list_3])
        mi_23_control = set(np.arange(list_length)[list_2 < list_3])
        mi_21_control = set(np.arange(list_length)[list_2 < list_1])
        mi_31_control = set(np.arange(list_length)[list_3 < list_1])
        mi_32_control = set(np.arange(list_length)[list_3 < list_2])
        mi_1_control = np.array(list(mi_12_control & mi_13_control))
        mi_2_control = np.array(list(mi_21_control & mi_23_control))
        mi_3_control = np.array(list(mi_31_control & mi_32_control))
        self.assertTrue((mi_1_test == mi_1_control).all(), "idx minimums do not match")
        self.assertTrue((mi_2_test == mi_2_control).all(), "idx minimums do not match")
        self.assertTrue((mi_3_test == mi_3_control).all(), "idx minimums do not match")
