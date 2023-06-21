# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:01:14 2023

@author: s.gorania
"""

import torch.nn as nn

model = nn.Sequential(
  nn.Linear(8, 12),
  nn.ReLU(),
  nn.Linear(12,20),
  nn.ReLU(),
  nn.Linear(20, 8),
  nn.ReLU(),
  nn.Linear(8, 1),
  nn.Sigmoid()
)
print(model)
