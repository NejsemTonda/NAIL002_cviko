
import numpy as np
from numpy.random import default_rng
from typing import List, Set, Dict, Tuple
from numpy.testing import assert_approx_equal, assert_allclose
rng = default_rng(2023) # common random number generator

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs, make_classification, make_moons


"""
# Assignment: $k$-Fold Cross-Validation

In this assignment, you will implement a function that compares two learning algorithms using $k$-fold cross-validation and apply it to compare some simple learning algorithms.

**Tasks:**
1. Implement class `Perceptron` for the perceptron learning algorithm. **(2 points)**
2. Implement function `cross_val` for evaluating the difference of errors between two given learning algorithms using $k$-fold cross-validation. **(4 points)**
3. Implement function `conf_interval` for computing confidence interval for an error difference when using $k$-fold cross-validation. **(1 point)**
4. By applying the functions implemented in the above tasks, compare several learning algorithms. **(3 points)**

This notebook was generated using `nbgrader`. It has a special format. Several cells contain tests. Some of the tests are hidden. Please, **do not break the format**:
1. *do not delete cells where you should insert your answers (code or text), and*
2. *do not copy complete cells (you can freely copy the contents of cells).
Otherwise, you can add and delete cells arbitrarily.* 
"""

"""
A learning algorithm tries to learn a target function $f: \\mathrm{R}^n \\to \\{0,1\\}$, where $\\mathrm{R}$ is the set of real numbers. The goal of a learning algorithm is to identify a function $h: \\mathrm{R}^n \\to \\{0,1\\}$, called *hypothesis*, from some class of functions $\\mathcal{H}$ such that the function $h$ is a good approximation of the target function $f$. The only information the algorithm can use is a sample $S\\subset X$ called a *training set*, and the correct value $f(x)$, called *label*, for all $x \\in S$. The sample $S$ is a set of $n$ elements from $X$ randomly selected according to some probabilistic distribution $\\mathcal{D}$.

We suppose that each learning algorithm is implemented as a subclass of the following 
class `LearningAlgorithm`:
"""

class LearningAlgorithm:
    """ Base class for learning algorithms. """

    def __init__(self, **learning_par):
        self.par = learning_par

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Learn the parameters of the algorithm. 
        The learned parameters are stored into member variables.
        """
        raise NotImplementedError
        
    def predict(self, X: np.ndarray) -> np.array:
        """ Apply the learned function to the input vectors `X`. 
        The returned value is a vector of the results - zeros and ones.
        """
        raise NotImplementedError
        
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """ Compute the mean accuracy on the test vectors `X`, where
        `y` is a vector (one-dimensional) containing 
        the correct labels for the vectors (the rows) in `X`. 
        """
        output = self.predict(X)
        return np.mean(output == y)

"""
where

Param      |Meaning
--------------|---------------------------------------------------------------
`learning_par`|is a dictionary of parameters of the learning algorithm,
`fit`         |is the learning function; it computes learned parameters and stores them into member variables; it can use the parameters from the member variable `self.par`,
`X`           |is a two-dimensional numpy array (training and test vectors are the rows of `X`),
`y`           |is a vector (a one-dimensional numpy array) of desired labels (zeros and ones), and
`predict`     |computes the learned function for the input vectors from a two-dimensional numpy array `X` (each row of the array is an input vector); it uses the learned parameters and/or the parameters from the member variable `self.par`; the returned value is a vector of the results - zeros and ones,
`score`       |computes the mean accuracy of the trained algorithm on the test vectors `X`, where `y[i]` contains the correct label for the vector (the row of `X`) `X[i]`.
"""

"""
## A simple learning algorithm `Memorizer`

Below, we implement a simple learning algorithm called `Memorizer` that memorizes all training samples and their true labels. Trained `Memorizer` answers 
* correctly on the inputs from the training set, and 
* randomly otherwise &ndash; it outputs zeroes and ones with the same ratio as in the training set.
"""

class Memorizer(LearningAlgorithm):
    
    def __init__(self, rng_seed=42):
        super().__init__(_seed=rng_seed, _rng=default_rng(rng_seed))
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._learned_par = [np.atleast_2d(X), y]
    
    def predict(self, X: np.ndarray) -> np.array:
        # generate random outputs
        # the following line of code ensures that each call of predict on the same `X` 
        # will return the same output; for a "serious" application, this line should be omitted
        self.par['_rng'] = default_rng(self.par['_seed'])
        
        X = np.atleast_2d(X)
        out = self.par['_rng'].binomial(1, self._learned_par[1].mean(), X.shape[0])
        for i in range(X.shape[0]):
            res = (self._learned_par[0] == X[i]).all(axis=1).nonzero()[-1]
            # print(f"{(self._learned_par[0] == X[i])=}")
            # print(f"{(self._learned_par[0] == X[i]).all(axis=1)=}")
            # print(f"{(self._learned_par[0] == X[i]).all(axis=1).nonzero()=}")
            # print(f"{res=}")
            if res.size > 0:
                out[i] = self._learned_par[1][res[0]]
        return out

"""
The above class can be used as follows:
"""

from pprint import pprint

X_train = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
y_train = np.array([1, 1, 0, 1, 0])

X_test = np.vstack((X_train, np.arange(0.1, 4.01, 0.1).reshape(-1, 2)))
y_test = np.hstack((y_train, np.zeros(X_test.shape[0] - y_train.shape[0])))

m = Memorizer()
m.fit(X_train, y_train)

# trained memorizer can predict the label for a sample represented as a list
X_list = [0.0, 1.0]
print(f'The prediction for input list {X_list} is {m.predict(X_list)[0]}')

# trained memorizer can predict the label for a sample represented as a numpy array
X_array = np.array([0.0, 0.0])
print(f'The prediction for input array {X_array} is {m.predict(X_array)[0]}')

prediction = m.predict(X_test)
print(f'par: {m.par}\nprediction: {prediction} positive predictions: {prediction.sum()}' 
      f'\nscore: {m.score(X_test, y_test)}')

# now we change the seed for the random number generator
m = Memorizer(rng_seed=2023)
m.fit(X_train, y_train)

prediction = m.predict(X_test)
print(f'par: {m.par}\nprediction: {prediction} positive predictions: {prediction.sum()}'
      f'\nscore: {m.score(X_test, y_test)}')


"""
## Task 1: Implement perceptron (2 points)

Similarly, we can adapt our implementation of perceptron learning algorithm from a previous lab.
Complete the implementation below. The extended weight vector will be initialized as 
`np.asarray(init_weights, dtype=float)` &ndash; this enables that `init_weights` 
can be either list of floats, or numpy array, and the empty weigth vector 
can be tested using `self.par['weights'].size == 0)`.
"""

class Perceptron(LearningAlgorithm):
    
    def __init__(self, init_weights=[], lr=1.0, max_epochs=1000):
        # Perceptron constructor
        # `init_weights` are the initial weights (including the bias) of the perceptron
        # `lr` is the learning rate
        # `max_epochs` is the maximum number of epochs
        super().__init__(weights = np.asarray(init_weights, dtype=float), lr=lr, max_epochs=max_epochs)
        
    def predict(self, X: np.ndarray) -> np.array:
        # Compute the output of the perceptron.
        # Input `X` can be
        #  * a vector, i.e., one sample
        #  * or a two-dimensional array, where each row is a sample.
        # Returns
        #  * a vector with values 0/1 with the output of the perceptron 
        #    for all samples in `X`
        # Raises an exception if the weights are not initialized.
        ### BEGIN SOLUTION
        if not self.par['weights'].size:
            raise Exception("Perceptron.predict: Weight vector was not initialized")
        X = np.atleast_2d(X)
        extended_X = np.hstack((X, np.ones((X.shape[0],1))))
        return (np.dot(extended_X, self.par['weights'].T) >= 0).astype(int)
        ### END SOLUTION
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray, lr=1.0) -> None:
        # perform one epoch perceptron learning algorithm 
        # on the training set `X` (two-dimensional numpy array of floats) with 
        # the desired outputs `y` (vector of integers 0/1) and learning rate `lr`.
        # If self.weights is empty, the weight vector is generated randomly.
        ### BEGIN SOLUTION
        X = np.atleast_2d(X)
        if not self.par['weights'].size:
            self.par['weights'] = np.random.default_rng().normal(size=(X.shape[1] + 1))
            print("Weights initialized in partial_fit")
        for i in range(X.shape[0]):
            pred = self.predict(X[i])
            self.par['weights'] += (y[i] - pred) * lr * np.hstack((X[i],1))   
            # print(f"After pattern {X[i]} weihts {self.par['weights']}")
        ### END SOLUTION
            
    def fit(self, X: np.ndarray, y: np.ndarray, lr=None, max_epochs=None) -> int:
        # trains perceptron using perceptron learning algorithm
        # on the training set `X` (two-dimensional numpy array of floats) with 
        # the desired outputs `y` (vector of integers 0/1). 
        # If `self.par['weights'] is empty, the weight vector is generated randomly.
        # If the learning rate `lr` is `None`, `self.par['lr']` is used.
        # If `max_epochs` is `None`, `self.par['max_epochs']` is used. 
        # Returns the number of epochs used in the training (at most `max_epochs`).
        ### BEGIN SOLUTION
        X = np.atleast_2d(X)
        if not self.par['weights'].size:
            self.par['weights'] = np.random.default_rng().normal(size=(X.shape[1] + 1))
            print(f"Weights initialized in fit {self.par['weights']=}")
        if max_epochs == None:
            max_epochs = self.par['max_epochs']
        if lr == None:
            lr = self.par['lr']
        epoch = 0
        while epoch < max_epochs and self.score(X, y) < 1.0:
            self.partial_fit(X, y, lr)
            print(f"After epoch {epoch}:, weights {self.par['weights']}, score {self.score(X,y)}")
            epoch += 1
        return epoch
        ### END SOLUTION

"""
A perceptron without weights cannot make predictions and must throw an exception (**0.5 points for this part**).
"""

p = Perceptron([1,2,3], max_epochs=10)
print(p.predict(np.array([1,1])))

p = Perceptron(max_epochs=10)
try:
    print(p.predict(np.array([1,1])))
except Exception as e:
    pass
else:
    raise AssertionError("Perceptron.predict with empty weights did not raise an exception")
 
### BEGIN HIDDEN TESTS
errors = []

# test that partial_fit correctly initializes weights if weights is not initialized 
# before calling partial_fit (X is a two-dimensional) 
p = Perceptron(max_epochs=30)
assert p.par['weights'].size == 0
X = np.array([[1,1], [0,1], [0,0]])
y = np.array([0, 1, 0])
p.partial_fit(X, y, lr=0.1)
if not p.par['weights'].size == X.shape[1] + 1:
    errors.append('ERROR: weights not initialized correctly in partial_fit when X is a two-dimensional array')

# test that fit correctly initializes weights if weights is not initialized 
# before calling fit (X is a two-dimensional array) 
p = Perceptron(max_epochs=50)
assert p.par['weights'].size == 0
X = np.array([[1,1], [0,1], [0,0]])
y = np.array([0, 1, 0])
p.fit(X, y, lr=0.1)
if not p.par['weights'].size == X.shape[1] + 1:
    errors.append('ERROR: weights not initialized correctly in fit when X is a two-dimensional array')

if errors != '':
    print(errors)
assert errors == []
### END HIDDEN TESTS

"""
Test the implementation of `Perceptron` carefully. It should pass all the tests below and some additional hidden tests (**1.5 points for this part**).
"""

p = Perceptron(init_weights=[1,-2,1], max_epochs=10)
assert (p.predict(np.array([[6,3],[5,3],[1,1],[1,1.00001]])) == [1,1,1,0]).all()

X_array = np.array([0.0, 0.0])
print(f"Prediction for the input array {X_array} is {p.predict(X_array)[0]}")
assert (p.predict(X_array)[0] == 1)

p = Perceptron(init_weights=[1,1,1], max_epochs=10)
X_train = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y_train = np.array([ 1, 0, 1, 0])

epochs = p.fit(X_train, y_train)
print(f"Training required {epochs} epoch(s)")
assert epochs == 2
print(f"Trained perceptron {p.par}")
assert (p.par['weights'] == np.array([ 0., -1.,  0.])).all()
print(f"Score on the training set {p.score(X_train, y_train)}")
assert_approx_equal(p.score(X_train, y_train), 1.0)

X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, random_state=1234)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
p = Perceptron([-1,1,2],0.5,5)
epochs = p.fit(X_train, y_train)
assert epochs==5
assert_approx_equal(p.score(X_test, y_test), 0.86666666)
assert_allclose(p.par['weights'], [0.04921529, 1.30631335, 0.        ])

### BEGIN HIDDEN TESTS
X_train = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y_train = np.array([0, 1, 0, 1])

X_test = np.arange(50).reshape(-1, 2)
y_test = np.ones(X_test.shape[0])

p = Perceptron(init_weights=[1,-2,1], max_epochs=1)
p.fit(X_train, y_train)
assert_approx_equal(p.score(X_test, y_test), 1.0)
############

p = Perceptron(init_weights=[1,-20,1], max_epochs=1)
p.fit(X_train, y_train)

# print(p.par)
# print(p.predict(X_test))
assert_approx_equal(p.score(X_test, y_test), 0.0)
############

p = Perceptron(init_weights=[10,-10,1], max_epochs=1)
p.fit(X_test, y_test)

# print(p.par)
# print(p.predict(X_test))
assert_approx_equal(p.score(X_test, y_test), 0.96)
############

from sklearn.datasets import make_blobs

seed = 4322

X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed + 1)

p = Perceptron(init_weights=[1,-1,0], max_epochs=400)
#assert p.fit(X_train, y_train) == 10
p.fit(X_train, y_train)

# print(p.par)
# print(p.predict(X_test))
# print('y_test', y_test)
assert_approx_equal(p.score(X_test, y_test), 0.56)
### END HIDDEN TESTS

"""
## Task 2: Implement k-fold cross-validation (4 points)

Then, it is easy to implement the following function `cross_val` that estimates the difference between the error of the hypothesis
learned by a learning algorithm `learn_alg1` and the error of the hypotheses learned by a learning algorithm
`learn_alg2` using `k`-fold cross-validation on
patterns `X` with the desired outputs `y`. The function returns the estimated
difference of errors `delta` and estimated standard deviation `s` of this
estimator. **You should implement your own function using `numpy`, not use 
any implementation of cross-validation from any third-party library!**

* If the value of `shuffle` is `False`, then
  * the order of samples must not be changed before partitioning into folds for `k`-fold cross-validation, 
  * all folds should be continuous parts of `X`, and
  * the first fold (from the beginning of array `X`) must be used as the first test set, and so on, until the last fold is used as the last test set.
* If the value of `shuffle` is `True`, then 
  * the patterns from `X` should be assigned randomly into `k` folds (then, in general, calling `cross_val` repeatedly with the same parameters can result in different folds and different outputs).

_Notes:_ 
* _The sizes of the folds can differ by at most 1._
* _The function computes **estimates** of the error difference and standard deviation of the estimated difference of errors. Therefore, it should compute the **sample** standard deviation according to the formula $$\\sigma = \\sqrt{\\frac{1}{k-1}\\sum_{i=1}^k (\\delta_i - \\bar{\\delta})^2}$$ where $\\delta_i$ is the error difference if the $i$-th fold is used as the test set, and $\\bar{\\delta}$ is the average error difference on all folds._
* _Be aware that for each iteration of k-fold cross-validation, the learning algorithms must start from the same state of the learning algorithm. E.g., you can make a copy of a `learn_alg1` using `copy.deepcopy(learn_alg1)`._

"""

import copy

def cross_val(k: int, learn_alg1: LearningAlgorithm, learn_alg2: LearningAlgorithm, 
              X: np.ndarray, y: np.ndarray, shuffle: bool = True, verbose=0) -> Tuple[float, float]:
    '''Estimates the difference between errors and the standard deviation of the difference for
    two learning algorithms.
    
        delta, std = cross_val(k, learn_alg1, learn_alg2, X, y, shuffle, verbose)
                               
    Args:
        k:           The number of folds used in k-fold cross-validation.
        learn_alg1:  The first learning algorithm.
        learn_alg2:  The second learning algorithm.
        name_apply1: The name of the function for applying the first learned function.
        X:           (2-d numpy array of floats): The training set; samples are rows.
        y:           (vector of integers 0/1): The desired outputs for the training samples.
        shuffle: If True, shuffle the samples into folds; otherwise, do not shuffle
                 the samples before splitting them into folds.
        verbose: If verbose == 0, it prints nothing. 
                 If verbose == 1, prints errors of both algorithms for each fold as a tuple.
                 If verbose == 2, for each fold, it prints 
                             the parameters of the first trained algorithm,
                             the parameters of the second trained algorithm, and 
                             the errors of both algorithms as an array with two elements 
        
    Returns:
        delta: The estimated difference: the error rate of the first algorithm minus 
            the error rate of the second algorithm computed using k-fold cross-validation.
        std: The sample standard deviation of the estimated difference of errors.            
    '''
    ### BEGIN SOLUTION
    if shuffle:
        permuted_data = rng.permuted(np.c_[X,y.reshape((-1,1))])
        X1 = permuted_data[:,:-1]
        y1 = permuted_data[:,-1]
    else:
        X1, y1 = X, y
    split_indices = np.array_split(np.arange(X1.shape[0]), k)
    err_dif = []
    for i in range(k):
        X_test = X1[split_indices[i]]
        y_test = y1[split_indices[i]]
        train_indices = np.concatenate(split_indices)
        train_indices = np.delete(train_indices, split_indices[i])
        X_train = X1[train_indices]
        y_train = y1[train_indices]
        a1 = copy.deepcopy(learn_alg1)
        a1.fit(X_train, y_train)
        err1 = 1 - a1.score(X_test, y_test)
        a2 = copy.deepcopy(learn_alg2)
        a2.fit(X_train, y_train)
        err2 = 1 - a2.score(X_test, y_test)
        if verbose == 2:
            print(a1.par)
            print(a2.par)
        if verbose >= 1:
            print(f"Errors at fold {i}:", np.array([err1, err2]))
        err_dif.append(err1 - err2)
    return np.mean(err_dif), np.std(err_dif, ddof=1)
        
    ### END SOLUTION

"""
In the next cell the implementatios of `cross_val` will be tested. The two visibe tests are followed with several hidden tests **(4 points for this part)**.
"""

X, y = make_moons(n_samples=200, noise=0.2, random_state=5)

delta, sigma = cross_val(5, Perceptron(init_weights=[1,-2,1], max_epochs=10), 
             Perceptron(init_weights=[1,-2,1], max_epochs=100), X, y, shuffle=False, verbose=1)
print(f"Error difference: {delta} std: {sigma}")
assert_allclose((delta, sigma), (0.015, 0.03354101967))


X, y = make_blobs(n_samples=600, centers=2, n_features=2, random_state=4)

delta, sigma = cross_val(10, Perceptron(init_weights=[1,1,1], max_epochs=10), 
             Perceptron(init_weights=[1,1,1], max_epochs=100), X, y, shuffle=False, verbose=1)
print(f"Error difference: {delta} std: {sigma}")
assert_allclose((delta, sigma), (-0.0133333333, 0.045677344))

### BEGIN HIDDEN TESTS
from sklearn.datasets import make_hastie_10_2

X, y = make_hastie_10_2(n_samples=300, random_state=42)
y = (y==1).astype(int)

delta, sigma = cross_val(5, Perceptron(init_weights=[1,-1,1,-1,1,-1,1,-1,1,-1,1], max_epochs=10), 
                         Perceptron(init_weights=[1,-1,1,-1,1,-1,1,-1,1,-1,1], max_epochs=100), 
                         X, y, shuffle=False, verbose=1)
print(f"Error difference: {delta} std: {sigma}")
assert_allclose((delta, sigma), (0.04, 0.0596285), rtol=0.001)

from sklearn.datasets import make_hastie_10_2

X, y = make_hastie_10_2(n_samples=300, random_state=42)
y = (y==1).astype(int)

delta, sigma = cross_val(10, Perceptron(init_weights=[1,-1,1,-1,1,-1,1,-1,1,-1,1], max_epochs=10), 
                         Perceptron(init_weights=[1,-1,1,-1,1,-1,1,-1,1,-1,1], max_epochs=100), 
                         X, y, shuffle=False, verbose=0)
print(f"Error difference: {delta} std: {sigma}")
assert_allclose((delta, sigma), (-0.0266667, 0.0750309), rtol=0.001)

from sklearn.datasets import make_hastie_10_2

X, y = make_hastie_10_2(n_samples=3000, random_state=42)
y = (y==1).astype(int)

delta, sigma = cross_val(5, Perceptron(init_weights=[1,-1,1,-1,1,-1,1,-1,1,-1,1], max_epochs=2), 
                         Perceptron(init_weights=[1,-1,1,-1,1,-1,1,-1,1,-1,1], max_epochs=10), 
                         X, y, shuffle=False, verbose=0)
print(f"Error difference: {delta} std: {sigma}")
assert_allclose((delta, sigma), (0.00966667, 0.00639010), rtol=0.001)

X, y = make_classification(n_samples=600, n_features=4, n_informative=2, 
                           n_redundant=1, n_repeated=1, random_state=4321)

delta, sigma = cross_val(5, Perceptron(init_weights=[1,-1,1,-1,1], max_epochs=2), 
                         Perceptron(init_weights=[1,-1,1,-1,1], max_epochs=10), 
                         X, y, shuffle=False, verbose=0)
print(f"Error difference: {delta} std: {sigma}")
assert_allclose((delta, sigma), (0.00666667, 0.0108653), rtol=0.001)

# test shuffle
delta_1shuffle, sigma1_shuffle = cross_val(5, Perceptron(init_weights=[1,-1,1,-1,1], max_epochs=2), 
                         Perceptron(init_weights=[1,-1,1,-1,1], max_epochs=10), 
                         X, y, shuffle=True, verbose=0)
delta_2shuffle, sigma2_shuffle = cross_val(5, Perceptron(init_weights=[1,-1,1,-1,1], max_epochs=2), 
                         Perceptron(init_weights=[1,-1,1,-1,1], max_epochs=10), 
                         X, y, shuffle=True, verbose=0)
assert (delta != delta_1shuffle or delta != delta_2shuffle) and \
        np.mean([delta, delta_1shuffle, delta_2shuffle]) < 0.3, \
        "The function cross_val does not implement shuffling correctly."
### END HIDDEN TESTS

"""
## Task 3: Implement computing of confidence interval (1 point)
When applying $k$-fold cross-validation, we will compute the confidence interval to estimate the difference of errors computed by the `cross_val()` function. Implement the following function.
"""

from scipy.stats import t

def conf_interval(delta: float, sigma: float, conf_level: float, k: int) -> Tuple[float,float]:
    """Compute confidence interval for the estimated difference of errors d 
    with standard deviation s returned from cross_val().
    
        low, high = conf_inteval(delta, sigma, conf_level, k)
    
    Args:
        delta: The difference of errors computed by k-fold cross-validation.
        sigma: The standard deviation of the difference of errors computed 
            by k-fold cross-validation.
        conf_level: The confidence level. A value between 0 and 1.
        k: The number of folds used in k-fold cross-validation.
    """
    ### BEGIN SOLUTION
    z = t.ppf((conf_level+1)/2, k-1)
    print("z ",z)
    return (delta - z * sigma / np.sqrt(k), delta + z * sigma / np.sqrt(k))    
    ### END SOLUTION

"""
Function `conf-interval` must pass the following and several hidden tests **(1 point for this part)**.
"""

assert_allclose(conf_interval(0.1, 0.03, 0.9, 10), (0.08260956,0.11739044))
assert_allclose(conf_interval(-0.1, 0.03, 0.95, 7), (-0.12774537,-0.07225463))
### BEGIN HIDDEN TESTS
assert_allclose(conf_interval(0.0, 0.05, 0.95, 7), (-0.04624229,0.04624229))
assert_allclose(conf_interval(-0.3, 0.06, 0.95, 5), (-0.37449984,-0.22550016))
assert_allclose(conf_interval(-0.3, 0.06, 0.99, 9), (-0.36710775,-0.23289225))
assert_allclose(conf_interval(0.08, 0.05, 0.95, 6), (0.02752822,0.13247178))
### END HIDDEN TESTS

"""
## Task 4: Compare learning algorithms using $k$-fold cross-validation (3 points)
<a id="compare_learn_alg"></a>

The above learning algorithms can be compared using the above function `cross_val` on the following datasets `dataset1` and `dataset2`.
"""

dataset1 = np.genfromtxt('Data1.txt', delimiter=' ', dtype = float)
print(f"{dataset1.shape=}")
print(dataset1[0])
dataset2 = np.genfromtxt('Data2.txt', delimiter=' ', dtype = float)
print(f"{dataset2.shape=}")
print(dataset2[0])

"""
The last column of both datasets are the correct labels (0 or 1). Hence, the dataset `dataset1` has 2-dimensional patterns and `dataset2` has 5-dimensional patterns. 
"""

"""
### Compare Perceptron with Memorizer (1.5 points)
Compare `Perceptron([1, 1, -1], 1, 20)` and `Memorizer(rng_seed=2023)` on `dataset1` using 5-fold cross-validation with the confidence level `0.95` and without shuffling (`shuffle=False`). 

After executing the next cell, the variables `low` and `high` should contain the lower and upper limits of the corresponding confidence interval, respectively, for the difference between the errors of the Perceptron and Memorizer.
"""

### BEGIN SOLUTION
import matplotlib.pyplot as plt

dataset = np.genfromtxt('Data1.txt', delimiter=' ', dtype = float)
X = dataset[:,:-1]
y = dataset[:,-1]
print('X[:5]\n', X[:5])
print('y[:5]\n', y[:5])
plt.scatter(X[:,0], X[:,1], c=y)

# we prepare two learning algorithms
alg1 = Perceptron([1, 1, -1], 1, 20)
alg2 = Memorizer(rng_seed=2023)

# run 5-fold cross-validation
k = 5
d, s = cross_val(k, alg1, alg2, X, y, shuffle=False, verbose=2)
print(f"Estimated error difference, standard deviation for the estimate: {np.array((d,s))}")

low, high = conf_interval(d, s, 0.95, k)
# print the interval to which the true error difference belongs 
# with the probability at least 95%
print(f"Confidence interval {np.array((low, high))}")
### END SOLUTION

# do not modify this cell
print(f"{low=} {high=}")

### BEGIN HIDDEN TESTS
assert_allclose(np.array((low, high)), (-0.46705781, -0.20627552))
### END HIDDEN TESTS

"""
Is the error difference between the above two learning algorithms statistically significant? Explain your answer! An answer 'YES' or 'NO' withut any explanation will be graded with 0 points.
"""

"""
Explain here!
"""

"""
### Compare two perceptrons (1.5 points)

Compare `Perceptron([1, -1, 1, -1, 1, -1], 1, 10)` and `Perceptron([1, -1, 1, -1, 1, -1], 1, 100)` on `dataset2` using 6-fold cross-validation with the confidence level `0.99` and without shuffling (`shuffle=False`). 

After executing the next cell, the variables `low` and `high` should contain the lower and upper limits of the corresponding confidence interval, respectively, for the difference between the errors of the two perceptrons.
"""

### BEGIN SOLUTION
dataset = np.genfromtxt('Data2.txt', delimiter=' ', dtype = float)
X = dataset[:,:-1]
y = dataset[:,-1]
print('X[:5]\n', X[:5])
print('y[:5]\n', y[:5])

# we prepare two learning algorithms
alg1 = Perceptron([1, -1, 1, -1, 1, -1], 1, 10)
alg2 = Perceptron([1, -1, 1, -1, 1, -1], 1, 100)

# run 5-fold cross-validation
k = 6
d, s = cross_val(k, alg1, alg2, X, y, shuffle=False, verbose=2)
print(f"Estimated error difference, standard deviation for the estimate: {np.array((d,s))}")

low, high = conf_interval(d, s, 0.99, k)
# print the interval to which the true error difference belongs 
# with the probability of at least 95%
print(f"Confidence interval {np.array((low, high))}")
### END SOLUTION

# do not modify this cell, it contains a hidden test

### BEGIN HIDDEN TESTS
assert_allclose(np.array((low, high)), (-0.0648525,   0.08151917))
### END HIDDEN TESTS

"""
Is the error difference between the above two learning algorithms statistically significant? Explain your answer! An answer 'YES' or 'NO' will be graded with 0 points.
"""

"""
Explain!
"""







"""
<a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=ec0476f8-4650-4c3a-9248-aafee9e966db' target="_blank">
<img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
"""
