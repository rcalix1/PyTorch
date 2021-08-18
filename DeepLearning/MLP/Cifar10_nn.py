## CIFAR 10 images
## neural net
## CIFAR consists of: 60,000 tiny 32x32 color RGB images
## labeled with integer 1 of 10 classes
## airplane (0), automobile (1), etc

## >> pip install torchvision

##########################################

import torch
import numpy as np
import imageio
import os
from matplotlib import pyplot as plt
import torch.optim as optim
import torch.nn as nn

from torchvision import datasets
from torchvision import transforms

###########################################


data_path = 'data/cifar10data/'
cifar10 = datasets.CIFAR10(data_path, train=True, download=True)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True)  ## gets the validation data

###########################################

print(     len(cifar10)     )

img, label = cifar10[99]

print(img)
print(label)

##plt.imshow(img)
##plt.show()

###########################################

print(   dir(transforms)   )

## .ToTensor()    converts numpy arrays or PIL images to tensor
## with correct dimensions (e.g. channel, height, width)

to_tensor = transforms.ToTensor()
img_t = to_tensor(img)
print(    img_t.shape   )

###########################################
## The transform can be passed directly to the
## entire dataset

tensor_cifar10  = datasets.CIFAR10(data_path, train=True, download=False,
                               transform=transforms.ToTensor()   )
                               
## now, these are tensors

img_t, _ = tensor_cifar10[99]
print(    type(   img_t   )      )
print(    img_t.shape       )
print(    img_t.dtype       )

## data is also scaled by ToTensor()
print(    img_t.min()    )
print(    img_t.max()    )

## to show images, we need to permutate them back
## e.g. from CxHxW   to    HxWxC

#plt.imshow(    img_t.permute(  1, 2, 0  )    )
#plt.show()

###########################################
## Normalize (e.g. zero mean and unitary standard deviation)

## First stack all tensors returned by the dataset along an extra dimension

imgs = torch.stack(     [img_t for img_t, _ in tensor_cifar10], dim=3      )
print(   imgs.shape   )

## compute mean and standard deviation
## view(3, -1) converts the images from 3x32x32 to 3x1024
## then the mean is taken over all 1024

view1 = imgs.view(3, -1).mean(dim=1)
print(   view1    )

view2 = imgs.view(3, -1).std(dim=1)
print(   view2    )

temp_transforms = transforms.Normalize(view1, view2)
print(  temp_transforms   )

###########################################

transformed_cifar10 = datasets.CIFAR10( data_path, train=True, download=False,
                                  transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(view1, view2)
                                  ])
                                  
                                  )
                                  

transformed_cifar10_val = datasets.CIFAR10( data_path, train=False, download=False,
                                  transform = transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize(view1, view2)
                                ])

                                 )


###########################################
## print normalized image

img_t, _ = transformed_cifar10[99]

##plt.imshow(    img_t.permute(1, 2, 0)    )
##plt.show()

###########################################
## Building the dataset
## Using only 2 of the 10 classes in CIFAR 10

label_map   =  {  0: 0,  2: 1    }
class_names =  ['airplane', 'bird']

cifar2 = [ (img, label_map[label])  for img, label in transformed_cifar10 if label in [0, 2]  ]

cifar2_val =  [  (img, label_map[label])  for img, label  in transformed_cifar10_val if label in [0, 2]  ]

###########################################
## the softmax function

def softmax(x):
    return torch.exp(x) / torch.exp(x).sum()


x = torch.tensor(    [1.0, 2.0,  3.0]     )

print(     softmax(x)      )

print(     softmax(x).sum()     )

###########################################
## softmax the Torch way

softmax = nn.Softmax(dim=1)

x1 = torch.tensor( [[1.0, 2.0, 3.0],
                   [1.0, 2.0, 3.0]])
                   
print(      softmax(x1)        )

###########################################
## now the model
## a fully connected neural net
## make vectors from 32x32x3   ->     1x3072


n_out = 2        ## two classes

model = nn.Sequential(
          nn.Linear(3072, 512),
          nn.Tanh(),
          nn.Linear(512, n_out),
          nn.Softmax(dim=1)
)

###########################################
## test the model without training
## weights are currently all random

img, _ = cifar2[0]
print(img)
print(img.shape)
#plt.imshow(   img.permute(1, 2, 0)    )
#plt.show()

###########################################
## from 3x32x32  to  1x3072


img_batch = img.view(-1).unsqueeze(0)
print(     img_batch.shape      )

###########################################
## predict with random weights

out = model(img_batch)
print(    out    )

###########################################
## torch.max() returns the max value and index along the designated dimension in the tensor

_, index = torch.max(    out, dim=1     )
print(     index      )

###########################################
## now the model with improvements
## a fully connected neural net
## make vectors from 32x32x3   ->     1x3072


n_out = 2        ## two classes

model2 = nn.Sequential(
          nn.Linear(3072, 512),
          nn.Tanh(),
          nn.Linear(512, n_out),
          nn.LogSoftmax(dim=1)         ## helps with optimization when probs close to zero
)

###########################################


model3 = nn.Sequential(
          nn.Linear(3072, 1024),
          nn.Tanh(),
          nn.Linear(1024, 512),
          nn.Tanh(),
          nn.Linear(512, 128),
          nn.Tanh(),
          nn.Linear(128, 2),
          nn.LogSoftmax(dim=1)         ## helps with optimization when probs close to zero
)

###########################################
## the loss NLL -> negative log likelihood
## book page 180

loss = nn.NLLLoss()

###########################################

img, label = cifar2[0]

out = model2(      img.view(-1).unsqueeze(0)       )

loss_val = loss(    out, torch.tensor([label])    )

print(     loss_val      )

###########################################
## data loader: batches, shuffling

train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)

###########################################

model_fn = model3

###########################################
## Now, training the classifier

learning_rate = 1e-2

optimizer = optim.SGD(  model_fn.parameters(), lr=learning_rate      )

loss_fn = nn.NLLLoss()

n_epochs = 100

for epoch in range(n_epochs):
    for imgs, labels in train_loader:              ## imgs is [64x3x32x32]
        batch_size = imgs.shape[0]
        outputs = model_fn(     imgs.view(batch_size, -1)       )
        loss = loss_fn( outputs, labels   )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

###########################################
## after training, now testing

val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=False)

correct = 0
total   = 0

with torch.no_grad():
    for imgs, labels in val_loader:
        batch_size = imgs.shape[0]
        outputs = model_fn(   imgs.view(batch_size, -1)   )
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int(       (predicted==labels).sum()       )
    
print("Accuracy: %f", correct/total)


###########################################

print("<<<<<<<<<<<<DONE>>>>>>>>>>>>>>")
