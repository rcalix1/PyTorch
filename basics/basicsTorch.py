## Basics of PyTorch
## >>conda install h5py

import torch
import numpy as np

import h5py


#######################################

a = torch.ones(3)

print(a)

print(    float(   a[1]   )      )

a[2]  =  17.0

print(  a  )

#########################################

b = [4.0, 1.0, 5.0, 3.0, 2.0, 1.0]
b_t = torch.tensor(     b       )
print(  b_t   )


c = [[4.0,1.0],[5.0,3.0],[2.0,1.0]]
c_t = torch.tensor(   c   )
print(   c_t    )

###########################################

print(   c_t.shape    )

###########################################

d_t = torch.zeros(3,2)
print(     d_t     )
d_t[0, 1] = 17.0
print(   d_t   )
print(    d_t[0]     )
print(    d_t[1:, :]      )

###########################################
## unsqueeze
## adds a dimension of size 1, just like unsqueeze


d_t_1 = d_t[..., None]
print(   d_t_1   )
print(   d_t_1.shape   )

###########################################

batch_t = torch.randn(2, 3, 5, 5)     ## shape (batch, channels for RGB, rows, columns)
print(batch_t)

###########################################
## broadcasting

## book page 47

###########################################
## data type for a tensor

double_points = torch.ones(10, 2, dtype=torch.double)
short_points  = torch.tensor(   [[1, 2],[3, 4]]  ,  dtype=torch.short    )

print(short_points)
print(   short_points.dtype   )

###########################################
###########################################
###########################################
###########################################
## The Tensor API

a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)

print(a.shape)
print(a_t.shape)

###########################################

a = torch.ones(3, 2)
a_t = a.transpose(0, 1)

print(a.shape)
print(a_t.shape)

###########################################
## transposing in higher dimensions

some_t = torch.ones(3, 4, 5)
transpose_t = some_t.transpose(0, 2)

print(some_t.shape)

print(transpose_t.shape)

###########################################
## numpy interoperability

points = torch.ones(3, 4)
points_np = points.numpy()
print(   points_np    )

points_t = torch.from_numpy(  points_np  )

###########################################
## serializing and saving to file

torch.save(points, 'data/example_points.t')

points2 = torch.load('data/example_points.t')

###########################################
## serializing with HDF5

f = h5py.File('data/points.hdf5', 'w')
dset = f.create_dataset('coords', data=points.numpy())
f.close()

###########################################

f = h5py.File('data/points.hdf5','r')
dset = f['coords']
last_points = dset[-2:]     ## loads just last  2 points

print(dset)
print(last_points)

last_points_t = torch.from_numpy(  last_points  )    ## now make it into a torch tensor
print(last_points_t)

f.close()

###########################################

print("<<<<<<<<<<<<DONE>>>>>>>>>>>>>>")
