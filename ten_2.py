import tensorflow as tf
from numpy.random import RandomState

rdm = RandomState(1)
X = rdm.rand(128,2)
Y = [[int (x0+x1 < 1)] for (x0,x1) in X]


