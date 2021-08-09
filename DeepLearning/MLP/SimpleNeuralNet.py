## simple neural net 

##########################################

import torch
import numpy as np
import imageio
import os
from matplotlib import pyplot as plt
import torch.optim as optim
import torch.nn as nn


###########################################

t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]

t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]

t_c = torch.tensor(  t_c   ).unsqueeze(1)   ## makes into column vector
t_u = torch.tensor(  t_u   ).unsqueeze(1)   ## makes into column vector

print(t_c.shape)
print(t_c)

###########################################
## splitting the dataset

n_samples = t_u.shape[0]
n_val     = int(     0.2 * n_samples      )

shuffled_indeces = torch.randperm(   n_samples    )

train_indeces = shuffled_indeces[      :-n_val]
val_indeces   = shuffled_indeces[-n_val:      ]

print(   train_indeces   )
print(   val_indeces     )

train_t_u = t_u[train_indeces]
train_t_c = t_c[train_indeces]

val_t_u = t_u[val_indeces]
val_t_c = t_c[val_indeces]

train_t_u_norm = 0.1 * train_t_u
val_t_u_norm   = 0.1 * val_t_u


###########################################

'''
fig = plt.figure()
plt.xlabel("temperature fahrenheit")
plt.ylabel("temperature celsius")
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.show()
'''

    
###########################################


def training_loop(n_epochs, optimizer, model, loss_fn, train_t_u, val_t_u, train_t_c, val_t_c):
    for epoch in range(1, n_epochs + 1):
     
        train_t_p = model(train_t_u)
        train_loss = loss_fn(train_t_p, train_t_c)
       
        val_t_p = model(val_t_u)
        val_loss = loss_fn(val_t_p, val_t_c)
        
        optimizer.zero_grad()
        train_loss.backward()   ## notice we only back prop on train_loss (not val_loss)
        optimizer.step()
        
        if epoch <= 1 or epoch % 1000 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss.item():.4f},"
                  f" Validation loss {val_loss.item():.4f}")
        
    
    
###########################################
## architecture

def model_fn_linear():
    return nn.Linear(1, 1)

###########################################

def model_fn_nn_1hidden():
    seq_model = nn.Sequential(
                    nn.Linear(1, 13),        ## 13 neurons in the hidden layer
                    nn.Tanh(),
                    nn.Linear(13, 1)         ## 1 output neuron
                )
    print(   seq_model   )
    return seq_model

###########################################

##model = model_fn_linear()
model = model_fn_nn_1hidden()

optimizer = optim.SGD(
       model.parameters(),
       lr=1e-3  )

## print(list(     linear_model.parameters()     ))

###########################################

training_loop(
        n_epochs = 3000,
        optimizer=optimizer,
        model = model,
        loss_fn = nn.MSELoss(),
        train_t_u = train_t_u_norm,
        val_t_u   = val_t_u_norm,
        train_t_c = train_t_c,
        val_t_c   = val_t_c
        )

###########################################

print(  [param.shape for param in model.parameters()  ]  )
print(  [param for param in model.parameters()  ]  )

for name, param in model.named_parameters():
    print('***********************************')
    print(   name, param.shape, param   )
    print('***********************************')

###########################################
## predict

print("pred", model(val_t_u_norm)   )
print("real", val_t_c)


###########################################

print("<<<<<<<<<<<<DONE>>>>>>>>>>>>>>")
