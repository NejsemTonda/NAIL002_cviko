
import numpy as np
import matplotlib.pyplot as plt

"""
# Generating Test Data


When we implement or test various methods for machine learning (e.g., neural networks), we need the ability to generate data with prescribed properties, "random" data, or randomly select data from a given set of data. To generate random data, we need to know the probability distribution of the data.


For a random variable $V$ with the *uniform probability distribution* on an interval $\\langle A,B\\rangle$ it holds:

* the density function of the distribution is $\\displaystyle h(x) = \\begin{cases}
             \\frac{1}{B-A} & \\text{for } x \\in \\langle A,B \\rangle\\\\
             0              & \\text{otherwise}
         \\end{cases}$

* a random number from the distribution can be generated using the function `random.uniform(A, B)`, which is included in the standard Python library
"""

import random

A, B = 10, 20
random.uniform(A,B)

"""
* An $m \\times n$ matrix `s` represented as `numpy` array of numbers from the uniform distribution on the interval $\\langle A, B \\rangle$ can be generated as
"""

A = 10
B = 20

m = 3
n = 4

# using just the Python standard library random
# YOUR CODE HERE
raise NotImplementedError()
print(s)

# using numpy
# YOUR CODE HERE
raise NotImplementedError()
print(s)

"""
For a random variable $V$ with the *normal (or Gauss) distribution* with mean $\\mu$ and variance $\\sigma^2$ it holds:

* the density function is the following $\\displaystyle p(x) = \\frac{ 1}{ \\sqrt{2 \\pi \\sigma^2}} e^{
  -\\frac{(x-\\mu)^2}{2\\sigma^2}}$

* For the random variable $V$ it holds that with the probability 95\\% its value is from the interval
  $\\langle\\mu-1.96\\sigma,\\mu+1.96\\sigma\\rangle$.


The function `random.gauss(mu, sigma)` generates a random number from the normal distribution with mean `mu` and variance `sigma`. Such distribution is denoted as ${\\cal N}(\\texttt{mu}, \\texttt{sigma})$.
"""

random.gauss(5,1)

"""
 A numpy array `s` of size $m \\times n$ with numbers from the normal distribution with mean `mu` and variance `sigma` can be generated as follows:
"""

mu, sigma = 2, 0.3 # mean and standard deviation

s = np.random.normal(mu, sigma, 10000)
print('s.shape:', s.shape)
print(s[:10])

"""
Verify the mean and the variance:
"""

# Is the mean equal to mu?
abs(mu - np.mean(s))

# is the standard deviation sigma?
abs(sigma - np.std(s, ddof=1))

"""
**Question:** Why in the above code `ddof=1` is used?
"""

"""
Let us display the histogram of the samples, along with the probability density function:
"""

rng = np.random.default_rng(12345)
s = rng.normal(mu, sigma, 10000)
bins = 30

plt.figure(figsize=(7,3))

# histogram
count, bins, ignored = plt.hist(s, bins, density=True, label='Histogram')

# theoretical density function
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r', label='Probability density function')

# estimated density functon
sample_mu = s.mean()
sample_sigma = s.std()
plt.plot(bins, 1/(sample_sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - sample_mu)**2 / (2 * sample_sigma**2) ),
         linewidth=2, color='g', label='Estimated probability density function')
_ = plt.legend()

?plt.figure

"""
**Question:** Why `density=True` is used in the above call to `plt.hist`?
"""

"""
Generate a two-dimensional array of samples from ${\\cal N}(3, 2.5)$:
"""

s = np.random.normal(3, 2.5, size=(2, 4))
print('s.shape:', s.shape)
print(s)

"""
## Generating Vector with Two Clusters

Implement a function `randv2n(n1,mu1,sigma1,n2,mu2,sigma2)` that
generates a one-row vector containing `n1` numbers
from the **normal distribution** with mean `mu1` and standard deviation
`sigma1` and another `n2` numbers from the normal distribution with mean `mu2` and standard deviation `sigma2`.

E.g., `randv2n(3,-10,1,4,10,1)` can return numbers (rounded)

`[  9.95050222 -10.61370077   9.87080321  -7.90992686   9.34644975
   8.75197336 -11.08548289]`

Note that the numbers from the two clusters are **randomly mixed** in the returned vector. This can be achieved using the function `random.shuffle(x)` for an iterable `x` or `np.random.shuffle` for numpy arrays. 
"""

def randv2n(n1,mu1,sigma1,n2,mu2,sigma2):
    # YOUR CODE HERE
    raise NotImplementedError()
    
print(randv2n(3,-10,1,4,10,1))

"""
Demonstrate your implementation by constructing a histogram of the returned vector. How can we set up the number of bins in the histogram?
"""

rv = randv2n(30,-10,1,40,10,1)
print(rv)

# plot the histogram of the vector rv
# YOUR CODE HERE
raise NotImplementedError()

"""
## Generating Clusters in 2D

Implement a function `randn2d(n, mu1, sigma1, mu2, sigma2, draw)`, with parameters `n`, `mu1`, `sigma1`, `mu2`, `sigma2` that are `numpy` vectors of the same length $n$. The function should generate a matrix of size $\\displaystyle \\left(\\sum_{i=1}^{n} \\texttt{n[i]}\\right) \\times 2$ containing random points in 2D. Each row of the resulting matrix is interpreted as the coordinates of a point on a plane. In the resulting matrix, For each `i`$=1,\\dots,n$, `n[i]` is the number of points in the `i`-th cluster generated randomly from the normal distribution with mean `mu1[i]` and standard deviation `sigma1[i]` in the first coordinate, and with mean `mu2[i]` and standard deviation `sigma2[i]` in the second coordinate.

If the function is called with `draw != None`, the function plots a graph with graphically distinguished clusters before returning the resulting matrix (e.g., using the function `plt.scatter()`.
"""

def randn2d(n, mu1, sigma1, mu2, sigma2, draw):
    # generate a random array with 2 columns
    # n[i] is the number of rows from cluster i
    # the first coordinate of elements from cluster i are generated from N(mu1[i],sigma1[i])
    # the second coordinate of elements from cluster i are generated from N(mu2[i],sigma2[i])
    # the retured array has shuffled rows
    # YOUR CODE HERE
    raise NotImplementedError()

res = randn2d([10,4,32], [1, 5, 20], [.3, .3, 2.1], [1, 11, 9], [.3, 0.1, 0.7], draw="yes")

"""
## Generating a Sample from Data

Let us write a function `selectk(x, k)` randomly selecting `k` rows from the input matrix `x`.
"""

def selectk(x, k):
    assert len(x) >= k
    rows = np.random.choice(x.shape[0], k, replace=False)
    return x[rows]
    pass

x = np.random.normal(3, 2.5, size=(10,3))

print(x)

"""
We can demonstrate our implementation on a randomly generated matrix with two columns. Plot the original and selected column vectors as points on a plane. Distinguish not selected and selected data with different colors or marks.
"""

selected = selectk(x,4)
print(selected)

# 3D scatter plot
# YOUR CODE HERE
raise NotImplementedError()

"""
The module `scikit-learn` contains a suitable function `train_test_split`, for randomly splitting data into training and test sets. How can we ensure that the splitting is done in the same way when we repeat the splitting?
"""

from sklearn.model_selection import train_test_split

train, test = train_test_split(x)
print("train:\n", train)
print("test:\n", test)

"""
We can use a parameter `random_state` to ensure that the splitting is done in the same way. 
"""

train, test = train_test_split(x, random_state=12345)
print("train:\n", train)
print("test:\n", test)

train, test = train_test_split(x, test_size=0.5, random_state=12345)
print("train:\n", train)
print("test:\n", test)


