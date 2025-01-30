"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable

class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.
        
        This function may not work correctly if there is infinite number of
        intersection points. 


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.
        """
        # Step 1: Sample points in the range [a, b]
        num_samples = 1000  # Large number for accuracy
        x_samples = np.linspace(a, b, num_samples)
        y_diff = f1(x_samples) - f2(x_samples)

        # Step 2: Find sign changes (potential intersection points)
        intersection_points = []
        for i in range(len(x_samples) - 1):
            if y_diff[i] * y_diff[i + 1] <= 0:  # Sign change detected
                x0, x1 = x_samples[i], x_samples[i + 1]

                # Step 3: Use bisection to refine the intersection
                while abs(x1 - x0) > (maxerr / 2) or abs(f1(xm) - f2(xm)) > (maxerr / 2):
                    xm = (x0 + x1) / 2
                    if (f1(xm) - f2(xm)) * (f1(x0) - f2(x0)) <= 0:
                        x1 = xm
                    else:
                        x0 = xm

                intersection_points.append((x0 + x1) / 2)

        return intersection_points

##########################################################################

import unittest
from sampleFunctions import *
from tqdm import tqdm

class TestAssignment2(unittest.TestCase):

    def test_sqr(self):
        ass2 = Assignment2()
        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])
        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):
        ass2 = Assignment2()
        f1, f2 = randomIntersectingPolynomials(10)
        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

if __name__ == "__main__":
    unittest.main()

'''
Question 2.1: Explain the key points in my implementation (4pt).

1. Sampled 1000 points in [a, b] using `np.linspace` and computed `f1(x) - f2(x)`.
2. Detected sign changes to identify potential intersections.
3. Used bisection to refine intersections until both:
   - `abs(x1 - x0) ≤ maxerr`
   - `abs(f1(xm) - f2(xm)) ≤ maxerr`
4. Efficient: O(1000) sampling + O(log(1/maxerr)) bisection per root.
'''
