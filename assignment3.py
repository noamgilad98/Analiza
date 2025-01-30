import numpy as np
import time
import random

class Assignment3:
    def __init__(self):
        """
        One-time setup before solving specific functions.
        """
        pass

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Numerically integrates f in [a, b] using at most n points.
        Uses the Trapezoidal Rule for efficiency and accuracy.
        """
        x = np.linspace(a, b, n, dtype=np.float32)
        y = np.array([f(xi) for xi in x], dtype=np.float32)
        dx = (b - a) / (n - 1)
        integral = np.sum((y[:-1] + y[1:]) * dx / 2)
        return np.float32(integral)

    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the enclosed area between two functions.
        If fewer than two intersections exist, returns NaN.
        """
        from Assignment2 import Assignment2

        ass2 = Assignment2()
        intersections = ass2.intersections(f1, f2, -100, 100, maxerr=0.001)

        if len(intersections) < 2:
            return np.float32(np.nan)

        area = np.float32(0.0)
        for i in range(len(intersections) - 1):
            a, b = intersections[i], intersections[i + 1]
            diff_func = lambda x: abs(f1(x) - f2(x))
            area += self.integrate(diff_func, a, b, 1000)

        return area

##########################################################################

import unittest
from sampleFunctions import *
from tqdm import tqdm

class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        r = ass3.integrate(f1, -1, 1, 10)
        self.assertEqual(r.dtype, np.float32)

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.09, 10, 20)
        true_result = -7.78662 * 10 ** 33
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

if __name__ == "__main__":
    unittest.main()
