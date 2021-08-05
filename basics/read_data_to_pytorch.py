## read data to pytorch
## >> conda install imageio

import torch
import numpy as np
import imageio
import os

###########################################

img_arr = imageio.imread('data/bobby.jpg')
print(   img_arr.shape   )


###########################################
## pytorch require images like this
## channel x height x width

img = torch.from_numpy( img_arr )
out = img.permute(  2, 0, 1  )
print(out.shape)

###########################################

batch_size = 3
batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)
data_dir = 'data/image-cats/'
filenames = [name for name in os.listdir(data_dir) if os.path.splitext(name)[-1] == '.png']

print(filenames)


for i, filename in enumerate(filenames):
    img_arr = imageio.imread(  os.path.join(data_dir, filename)  )
    img_t = torch.from_numpy( img_arr )
    img_t = img_t.permute(2, 0, 1)
    img_t = img_t[:3]                           ## keeps only the first 3 channels (no transparency, etc)
    batch[i] = img_t
    
    
###########################################
## normalize the batch

batch = batch.float()

n_channels = batch.shape[1]
for c in range(n_channels):
    mean = torch.mean(batch[:, c])
    std  = torch.std( batch[:, c])
    batch[:, c] = (batch[:, c] - mean) / std


###########################################
## unsqueeze

print(  batch.shape  )

batch_new = torch.unsqueeze(batch, 0)

print(  batch_new.shape   )

###########################################
## read csv
## white wine portugal

import csv
wine_path = "data/winequality-white.csv"
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)

print(    wineq_numpy     )
print(    wineq_numpy.shape     )
wineq = torch.from_numpy(  wineq_numpy  )
print(  wineq.shape   )
print(  wineq.dtype   )

###########################################
## get X and y

data = wineq[:, :-1]           ## X
target = wineq[:, -1]          ## y

print(  data.shape    )
print(  target.shape  )

###########################################
## one hot encoding

print(  target  )

target = target.long()

print(  target  )

target_onehot = torch.zeros(target.shape[0], 10)

## take values in target and makes them one hot encoded
target_onehot.scatter_(1, target.unsqueeze(1), 1.0)

print(    target_onehot    )

print(    target_onehot.shape    )

###########################################
## unsqueeze the y data for matrix multiplications

target_unsqueezed = target.unsqueeze(1)

print(  target.shape   )
print(  target_unsqueezed.shape   )

###########################################
## normalize the data X

data_mean = torch.mean(data, dim=0)

data_var = torch.var(data, dim=0)

data_normalized = (data - data_mean) / torch.sqrt(  data_var  )

###########################################
## Text

with open('data/1342-0.txt', encoding='utf8') as f:
    text = f.read()
    

lines = text.split('\n')
line = lines[200]
print(  line   )

letter_t = torch.zeros(   len(line), 128   )
print(  letter_t.shape   )

for i, letter in enumerate(line.lower().strip()):
    letter_index = ord(letter) if ord(letter) < 128 else 0
    letter_t[i][letter_index] = 1
    
print(  letter_t  )


###########################################

print("<<<<<<<<<<<<DONE>>>>>>>>>>>>>>")
