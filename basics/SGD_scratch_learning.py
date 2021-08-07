## basics of learning
## with a linear function

##########################################

import torch
import numpy as np
import imageio
import os
from matplotlib import pyplot as plt
import torch.optim as optim

##########################################

t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]

t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]

t_c = torch.tensor(  t_c   )
t_u = torch.tensor(  t_u   )

## t_u = 0.1 * t_u

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

## t_c = w * t_u + b

def model(t_u, w, b):
    return w * t_u + b
    
###########################################

def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c) ** 2
    return squared_diffs.mean()
    
###########################################
## now all previous things combinedin one function
## params.grad needs to be zero out every iteration
## a corky but useful aspect of pytorch ?

def training_loop(n_epochs, optimizer, params, train_t_u, val_t_u, train_t_c, val_t_c):
    for epoch in range(1, n_epochs + 1):
            
        ## this is one seperate train computational graph
        train_t_p = model(train_t_u, *params)
        train_loss = loss_fn(train_t_p, train_t_c)
        
        ## this is one seperate (other) val computational graph
        with torch.no_grad():                 ## switch off gradient since val not used for training
            val_t_p = model(val_t_u, *params)
            val_loss = loss_fn(val_t_p, val_t_c)
            assert val_loss.requires_grad == False
        
        ## note: the only tensors these 2 computational graphs (train and val)
        ## have in common are the parameters (params)
        
        optimizer.zero_grad()
        train_loss.backward()   ## notice we only back prop on train_loss (not val_loss)
        optimizer.step()
        
        if epoch <= 3 or epoch % 500 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss.item():.4f},"
                  f" Validation loss {val_loss.item():.4f}")
        
            
    return params
    
###########################################

params = torch.tensor(  [1.0, 0.0], requires_grad=True  )
learning_rate = 1e-2
optimizer = optim.SGD(  [params], lr=learning_rate  )
##optimizer = optim.Adam( [params],  lr=learning_rate   )

###########################################

result = training_loop(
        n_epochs = 3000,
        optimizer=optimizer,
        params = params,
        train_t_u = train_t_u_norm,
        val_t_u   = val_t_u_norm,
        train_t_c = train_t_c,
        val_t_c   = val_t_c
        )
        
###########################################
        
print(  result  )

###########################################

print("<<<<<<<<<<<<DONE>>>>>>>>>>>>>>")
