# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 10:56:43 2023

@author: s.gorania
"""

# Example of PyTorch library
import torch

# declare two symbolic floating-point scalars
a = torch.tensor(1.5)
b = torch.tensor(2.5)

# create a simple symbolic expression using the add function
c = torch.add(a, b)
print(c)
