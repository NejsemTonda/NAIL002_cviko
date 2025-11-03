
!pip install icecream==2.1.8

import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

"""
# Multi-Layered Neural Networks and the Backpropagation Algorithm
 
Let us take a neural network with the topology [2,2,1], i.e., the network
has 2 input neurons, 2 hidden neurons in a single hidden layer, and one
output neuron. Let the weights of synapses between the input and the
hidden layer be in the following matrix:
"""

w_i_h = np.array([[1, -1],
                  [3,  1]])

"""
`w_i_h[i,j]` is the weight of the synapse from the input `i` into the
hidden neuron `j`. I.e., each row of the weight matrix corresponds to
the weights of synapses leading **from** one neuron!

Let the synaptic weights between the hidden and the output layer
be in the matrix:
"""

w_h_o = np.array([[2], [-1]])

"""
`w_h_o[i,0]` is the weight of the connection from the hidden neuron `i` 
to the output neuron (with index 0). Thresholds of the hidden neurons are in the vector:
"""

 b_h = np.array([0, 1])

"""
and the threshold of the outout neuron is:
"""

b_o = np.array([-1])

"""
Hence the weights from the input layer into the hidden layer with added 
virtual neuron with fixed output 1 (for representing thresholds) are:
"""

# note that r_ is not a method of numpy array!
w_i_hb = np.r_[w_i_h, b_h.reshape(1,-1)]
w_i_hb

"""
The weights from the hidden layer into the output layer
with added virtual neuron with output 1 are:
"""

w_h_ob = np.r_[w_h_o, b_o.reshape(1,-1)]
w_h_ob

"""
A sigmoidal transfer function $$logsig(x) = \\frac{1}{1 + e^{-\\lambda x}}$$ can be implemented as
"""

def sigmoid(x, lam=1.0):
    # sigmoid transfer function
    #     sigmoid(x) = 1/(1 + exp{-lam * x)
    return 1 / (1 + np.exp(-lam * x))

1/(1+np.exp(-3))

sigmoid(3)

"""
This is the sigmoid function with the slope $\\lambda=$ `lam`. The default value for the slope is $\\lambda = 1$.
"""

"""
## Tasks

### Output of the network and its error

* *Let $\\lambda=1.5$. Compute the output of the network for the input patterns `p1` and `p2`.*
"""

lam = 1.5
p1 = np.array([-1, 1])
p2 = np.array([ 1,-2])

print(f"w_i_hb=\n{w_i_hb}")
print(f"{p1=}")
print("p1 extended")   # just print it
# YOUR CODE HERE
raise NotImplementedError()

print("potential of the first hidden neuron")   # just print it
# YOUR CODE HERE
raise NotImplementedError()

print("output of the first hidden neuron")   # just print it
# YOUR CODE HERE
raise NotImplementedError()

print("output of the second hidden neuron")   # just print it
# YOUR CODE HERE
raise NotImplementedError()

print("output of the (complete) hidden layer")
y_h = ...
# YOUR CODE HERE
raise NotImplementedError()
print(f"y_h=\n{y_h}")

print("output of the output layer")
y_o = ... 
# YOUR CODE HERE
raise NotImplementedError()
print(f"y_o=\n{y_o}")

print(f"w_i_hb=\n{w_i_hb}")
print(f"{p2=}")
print("p2 extended")   # just print it
# YOUR CODE HERE
raise NotImplementedError()

print("potential of the first hidden neuron")   # just print it
# YOUR CODE HERE
raise NotImplementedError()

print("output of the first hidden neuron")   # just print it
# YOUR CODE HERE
raise NotImplementedError()

print("output of the second hidden neuron")   # just print it
# YOUR CODE HERE
raise NotImplementedError()

print("output of the (complete) hidden layer")
y_h = ...
# YOUR CODE HERE
raise NotImplementedError()
print(f"y_h=\n{y_h}")

print("output of the output layer")
y_o = ... 
# YOUR CODE HERE
raise NotImplementedError()
print(f"y_o=\n{y_o}")

"""
* *Compute the output of the network for the whole training set `X` consisting of the patterns `p1` and `p2`.*
"""

X = np.vstack((p1,p2))
print(f"X=\n{X}")
print(np.c_[X, np.ones(X.shape[0])])
y_h = ...
# YOUR CODE HERE
raise NotImplementedError()
print(f"y_h=\n{y_h}")
y_o = ...
# YOUR CODE HERE
raise NotImplementedError()
print(f"y_o={y_o}")

"""
The input pattern  `p1` is a training vector with a desired
output of 0.9, and the input pattern `p2` is also a training pattern with a desired output of 0.8. Hence, the desired outputs can be stored in an array, where row `d[i]` is the desired output for the pattern `X[i]`.
"""

d = np.array([[0.9], [0.8]])
print(f"desired output d=\n{d}")
print(f"actual output y_o=\n{y_o}")

"""
* *What is the error of the network on each of the patterns `p1` and `p2`?*
"""

# YOUR CODE HERE
raise NotImplementedError()

"""
* *What is the mean squared error (MSE) of the network on the whole training set?*
"""

# YOUR CODE HERE
raise NotImplementedError()

"""
### Training

* *How will the weights of the network be changed after one step of the backpropagation learning algorithm (without momentum) with the training pattern `p1` and the learning rate $\\alpha = 0.2$?*
"""

alpha = 0.2

"""
The error terms at neuron $j$ in the output layer

$$\\hspace{4em} \\displaystyle \\delta_j = (d_j-y_j)\\cdot \\lambda  y_j (1 - y_j)$$
"""

# YOUR CODE HERE
raise NotImplementedError()

"""
The error term at neuron $j$ in a hidden layer
$$\\hspace{4em} \\displaystyle \\delta_j = \\big(\\sum_k \\delta_k w_{jk}\\big) \\cdot \\lambda y_j (1 - y_j)$$
"""

delta_h = ...               # delta terms at the hidden layer

w_h_ob1 = ...               # new weights from the hidden to the output layer
w_i_hb1 = ...               # new weights form the input to the hidden layer
# YOUR CODE HERE
raise NotImplementedError()

delta_o

###alternative ???
delta_h = ...               # delta terms at the hidden layer

w_h_ob1 = ...               # new weights from the hidden to the output layer
w_i_hb1 = ...               # new weights form the input to the hidden layer
# YOUR CODE HERE
raise NotImplementedError()

"""
   
* How will change the output of the network for input `p1` after the first 
  iteration of the backpropagation algorithm?*
"""

# YOUR CODE HERE
raise NotImplementedError()

"""
* *Estimate the number of iterations over the pattern `p1` necessary to obtain an error "close" to 0*
"""

alpha = 0.2
lam = 1.0



"""
**Notation:**

Using `numpy` for working with vectors and matrices when we train a neural network has some problems:
* Input: input patterns are stored as rows in a 2D matrix $X$, but one input pattern is a 1D vector.
* Output, desired output: output patterns are stored as rows in a 2D matrix $Y$, however one output pattern is a 1D vector.
* Output of hidden neurons: can be stored in rows of a 2D matrix if we compute output for more than one pattern, but it is a 1D vector if we compute with one input vector.

A possible solution: is to *store vectors as two-dimensional arrays*:
* Then we can distinguish row and column vectors.
* If we work with a single vector, we will convert it into a row vector.
"""

p1_2d = p1.reshape(1,-1)
print("p1_2d\n",p1_2d)

# output of the hidden neurons
y_h = ...
print("y_h\n", y_h)

# output of the network 
y_o = ...
print("y_o\n", y_o)

delta_o = ...
print("delta_o\n", delta_o)

"""
Note that `delta_o` **is a row vector**? Why?
"""

print("np.c_[y_h,[[1]]]\n", np.c_[y_h,[[1]]])

w_h_ob1 = w_h_ob + ...
print("w_h_ob1\n", w_h_ob1)

delta_h = ...
print("delta_h\n", delta_h)


w_i_hb1 = ...
print("w_i_hb1\n", w_i_hb1)

"""
Now for the second pattern `p2`.
"""

p2_2d = p2.reshape(1,-1)
print("p2_2d\n",p2_2d)

# output of the hidden neurons

y_h = ...
print("y_h\n", y_h)

y_o = ...
print("y_o\n", y_o)


delta_o = ...
print("delta_o\n", delta_o)

w_h_ob1 = ...
print("w_h_ob1\n", w_h_ob1)

delta_h = ...
print("delta_h\n", delta_h)

w_i_hb1 = ... 
print("w_i_hb1\n", w_i_hb1)



"""
<a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=ed9da5be-77ca-4c93-b711-b98404e1e8a3' target="_blank">
<img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
"""
