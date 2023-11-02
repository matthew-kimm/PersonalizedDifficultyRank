import numpy as np
import unittest
from partial_order import partial_order
from metrics import ap_score


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

        self.data_M = np.array([[10],
                                [20]])

        self.data_N = np.array([[11],
                                [21]])

        self.data_P = np.array([[10],
                                [10]])

        self.data_Q = np.array([[20],
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

        self.order_M = partial_order(self.data_M)
        self.order_N = partial_order(self.data_N)
        self.order_P = partial_order(self.data_P)
        self.order_Q = partial_order(self.data_Q)

        self.order_X = partial_order(self.data_X)
        self.order_Y = partial_order(self.data_Y)

        self.items = [0, 1, 2]
        self.items_Y = [0, 1, 2, 3]

    def test_match_order(self):
        self.assertTrue(ap_score(self.order_A, self.order_A) == 1)
        self.assertTrue(ap_score(self.order_N, self.order_M) == 1)

    def test_incorrect_orders(self):
        self.assertTrue(ap_score(self.order_A, self.order_B) == 3/4)
        self.assertTrue(ap_score(self.order_A, self.order_C) == 1/2)
        self.assertTrue(ap_score(self.order_A, self.order_D) == 1/4)
        self.assertTrue(ap_score(self.order_A, self.order_E) == 1/2)
        self.assertTrue(ap_score(self.order_A, self.order_F) == 0)
        self.assertTrue(np.isnan(ap_score(self.order_P, self.order_M)))
        self.assertTrue(ap_score(self.order_Q, self.order_M) == 0)

    def test_ties_orders(self):
        self.assertTrue(ap_score(self.order_A, self.order_G) == 0)
        self.assertTrue(ap_score(self.order_Y, self.order_X) == 1/3)
