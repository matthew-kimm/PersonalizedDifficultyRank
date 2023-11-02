import numpy as np
import unittest
from partial_order import partial_order
from metrics import ndpm_score


class TestAPScore(unittest.TestCase):
    def setUp(self) -> None:
        self.data_A = np.array([[30],
                                [20],
                                [10]])
        self.data_B = np.array([[30],
                                [10],
                                [20]])
        self.data_C = np.array([[20],
                                [30],
                                [10]])
        self.data_D = np.array([[10],
                                [30],
                                [20]])
        self.data_E = np.array([[20],
                                [10],
                                [30]])
        self.data_F = np.array([[10],
                                [20],
                                [30]])

        self.data_G = np.array([[10],
                               [10],
                               [10]])

        self.data_X = np.array([[10],
                                [10],
                                [20],
                                [5]])

        self.data_Y = np.array([[40],
                                [30],
                                [20],
                                [10]])

        self.order_A = partial_order(self.data_A)
        self.order_B = partial_order(self.data_B)
        self.order_C = partial_order(self.data_C)
        self.order_D = partial_order(self.data_D)
        self.order_E = partial_order(self.data_E)
        self.order_F = partial_order(self.data_F)
        self.order_G = partial_order(self.data_G)
        self.order_X = partial_order(self.data_X)
        self.order_Y = partial_order(self.data_Y)

    def test_match_order(self):
        self.assertTrue(ndpm_score(self.order_A, self.order_A) == 0)

    def test_incorrect_orders(self):
        self.assertTrue(ndpm_score(self.order_A, self.order_B) == 1/3)
        self.assertTrue(ndpm_score(self.order_A, self.order_C) == 1/3)
        self.assertTrue(ndpm_score(self.order_A, self.order_D) == 2/3)
        self.assertTrue(ndpm_score(self.order_A, self.order_E) == 2/3)
        self.assertTrue(ndpm_score(self.order_A, self.order_F) == 1)

    def test_ties_orders(self):
        self.assertTrue(ndpm_score(self.order_A, self.order_G) == 1/2)
        # Nan
        self.assertTrue(np.isnan(ndpm_score(self.order_G, self.order_A)))
        self.assertTrue(ndpm_score(self.order_X, self.order_Y) == 2/5)
        self.assertTrue(ndpm_score(self.order_Y, self.order_X) == 5/12)
