import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict
import numpy as np
from physicsnemo.sym.node import Node

class ComputeSin(nn.Module):
    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {"sin_x": torch.sin(in_vars["x"])}
node = Node(['x'], ['sin_x'], ComputeSin())

result = node.evaluate({"x": (torch.ones(10, 1))*np.pi/4,})
print(result)