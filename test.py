from abc import ABC, abstractmethod
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# debugging 1
class base(ABC):
    def __init__(self, a, b, c, *args):
        self.a = a
        self.b = b
        self.c = c

class A(base):
    def __init__(self, a, d, e, b, c):
        super().__init__(b=b, c=c) # debugging 1: error raised if a is missing in the argument
        self.d = d
        self.e = e
#a = A(1,4,5,2,3)
#print("a=%d, b=%d, c=%d" % (a.a, a.b, a.c, a.d, a.e))
