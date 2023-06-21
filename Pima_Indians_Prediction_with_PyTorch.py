# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:22:02 2023

@author: s.gorania
"""
# Import relevant packages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load and format data
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',', skiprows=1)
X = dataset[:, 0:8]
y = dataset[:, 8]
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# Build model
model = nn.Sequential(
  nn.Linear(8, 12),
  nn.ReLU(),
  nn.Linear(12, 20),
  nn.ReLU(),
  nn.Linear(20, 8),
  nn.ReLU(),
  nn.Linear(8, 1),
  nn.Sigmoid()
)
print(model)

# Define loss and optimiser functions
loss_fn = nn.BCELoss()  # binary cross-entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define no. of epochs and batch size
n_epochs = 100
batch_size = 10

# Use model
for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')

# Stating that the model is run for inference, so that gradient calculation is
# not required. This consumes less resources.
i = 5
X_sample = X[i:i+1]
model.eval()
with torch.no_grad():
    y_pred = model(X_sample)
print(f"{X_sample[0]} -> {y_pred[0]}")

# Evaluate the model and get the accuracy
model.eval()
with torch.no_grad():
    y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")
