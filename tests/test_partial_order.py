import numpy as np
import unittest
from partial_order import partial_order


class TestPartialOrder(unittest.TestCase):
    def setUp(self) -> None:
        self.item_count = 4
        self.ratings_1 = np.array([[10], [20]])
        self.ratings_2 = np.array([[5], [7], [10]])

        # partial orders are padded with 0 as a catch-all
        self.expected_po_1 = np.array([[0, -1, 0],
                                       [1,  0, 0],
                                       [0,  0, 0]])
        self.expected_po_2 = np.array([[0, -1, -1, 0],
                                       [1,  0, -1, 0],
                                       [1,  1,  0, 0],
                                       [0,  0,  0, 0]])

    def test_partial_orders(self):
        po_1 = partial_order(self.ratings_1)
        po_2 = partial_order(self.ratings_2)

        self.assertTrue(np.array_equal(po_1, self.expected_po_1))
        self.assertTrue(np.array_equal(po_2, self.expected_po_2))
