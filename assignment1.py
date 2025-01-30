"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random


class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        starting to interpolate arbitrary functions.
        """

        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time. 
        The assignment will be tested on variety of different functions with 
        large n values. 
        
        Interpolation error will be measured as the average absolute error at 
        2*n random points between a and b. See test_with_poly() below. 

        Note: It is forbidden to call f more than n times. 

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.** 
        
        Note: sometimes you can get very accurate solutions with only few points, 
        significantly less than n. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """
        x_values = np.linspace(a, b, n)
        y_values = np.array([f(x) for x in x_values])

        def interpolated_function(x):
            if x <= a:
                return y_values[0]
            if x >= b:
                return y_values[-1]
            # Find the interval x is in
            i = np.searchsorted(x_values, x) - 1
            x0, x1 = x_values[i], x_values[i + 1]
            y0, y1 = y_values[i], y_values[i + 1]
            # Linear interpolation
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

        return interpolated_function


##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)

if __name__ == "__main__":
    unittest.main()
'''
Question 1.1: Explain the key points in your implementation (4pt).

1. Sampling Function Values Efficiently
   - Used np.linspace(a, b, n) to generate n evenly spaced points in [a, b].
   - Evaluated f(x) at these points only once to stay within the function call limit.

2. Piecewise Linear Interpolation
   - Found the nearest interval for a given x using np.searchsorted.
   - Applied linear interpolation using the formula:
     g(x) = y0 + ((y1 - y0) / (x1 - x0)) * (x - x0).

3. Handling Edge Cases
   - If x <= a, return y0 (left boundary).
   - If x >= b, return yn (right boundary).

4. Time Complexity Considerations
   - Preprocessing: O(n) (sampling and function evaluations).
   - Query Time: O(1) (search and interpolation).
'''
