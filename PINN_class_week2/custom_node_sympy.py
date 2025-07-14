import torch
import numpy as np
from sympy import Symbol, sin
from physicsnemo.sym.node import Node

node = Node.from_sympy(sin(Symbol("x")), "sin_x")

result = node.evaluate({"x": (torch.ones(10, 1))*np.pi/4,})
print(result)