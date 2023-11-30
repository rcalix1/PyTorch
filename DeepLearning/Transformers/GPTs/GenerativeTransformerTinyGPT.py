# ## A small ChatGPT style Transformer
# * concepts in NLP and Transformers 
# * generative mdoels

############################################################ 

import torch
import numpy as np
import requests
## import tiktoken
import torch.nn as nn

from torch.nn import functional as F

############################################################

## !pip install requests
## !pip install tiktoken    ## requires python   >    3.9

############################################################

torch.manual_seed(1337)

block_size = 256      ## max content length for predictions
batch_size = 64 
max_iters  = 5000
eval_interval = 500
learning_rate = 3e-4             ## 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
vocab_size = 65
n_embd  = 384                  ## every id gets embedded to vector of this size
n_head  = 6
n_layer = 6
dropout = 0.2

############################################################

input_file_path = 'input.txt'

## data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

with open(input_file_path, 'r', encoding='utf-8') as f:
    text = f.read()
    
############################################################

print("length of data in characters")
len(text)

############################################################

chars = sorted(     list(set(text))   )

vocab_size = len(chars)

print(  ''.join(chars)  )

############################################################# 
## tokenizer

stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }

encode = lambda s: [ stoi[c]          for c in s   ]    ## encoder: string to integer
decode = lambda l: ''.join(   itos[i] for i in l   )    ## decoder: interger to string

#############################################################

data = torch.tensor(   encode(text), dtype=torch.long   )
n    = int(   0.9*len(data)   )
train_data = data[:n]
val_data   = data[n:]

#############################################################

def get_batch(split):
    if split == "train":
        data = train_data
    else:
        data = val_data
    ix = torch.randint(   len(data) - block_size, (batch_size,)   )
    x  = torch.stack(    [  data[ i : i+block_size ]   for i in ix]    ) 
    y  = torch.stack(    [  data[ i+1 : i+1+block_size ]   for i in ix]    )
    
    x, y = x.to(device), y.to(device)

    return x, y

############################################################

@torch.no_grad()    ## for efficiency
def estimate_loss():
    out = {}
    model.eval()   ## no training
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  ## back to training
    return out

##########################################################################################


class Head(nn.Module):
    """ one head of self-attention """
    
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        ## the mask tril is not part of the graph since only for masking
        ## so register buffer makes it a thing out of the graph
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)              ## (B, T, C)
        q = self.query(x)            ## (B, T, C)
        
        wei = q @ k.transpose(-2, -1) * C**-0.5       ## (B, T, C) @ (B, C, T)  -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))     ## (B, T, T)
        wei = F.softmax(wei, dim= -1)           ## (B, T, T)
        wei = self.dropout(   wei   )
        
        ## perform the weighted aggregation of the values
        v   = self.value(  x  )   ## (B, T, C)
        out = wei @ v             ## (B, T, T) @ (B, T, C) -> (B, T, C)
        
        return out
        
##########################################################################################


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(  [Head(head_size) for _ in range(num_heads) ] )
        self.proj  = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat(   [ h(x) for h in self.heads], dim = -1   )
        out = self.proj(  out   )
        out = self.dropout(   out   )
        return out

##########################################################################################

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)

##########################################################################################

class Block(nn.Module):
    """ Transformer block: comuunication followed by computation """
    
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward( n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        ## these normalizations (ln1, ln2) are about the only thing different from
        ## the original Vaswani paper. In the paper, they are done at the end of forward
        ## but now they are usually done at the beginning of forward
        x = x + self.sa(     self.ln1(x)      )
        x = x + self.ffwd(   self.ln2(x)      )
        return x
    
##########################################################################################


class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)     ## positional encoding 
        self.blocks = nn.Sequential(
                *[   Block(n_embd, n_head=n_head) for _ in range(n_layer)    ]
        )
        self.ln_f    = nn.LayerNorm(  n_embd    )        ## final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    
    def forward(self, idx, targets=None):
        
        B, T = idx.shape
        
        ## ids and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)      ## batch, time, embed (4, 8, 32) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))      ## (T, C)
        x = tok_emb + pos_emb    ## (B, T, C)
        x = self.blocks(  x  )   ## (B, T, C)        
        x = self.ln_f(x)         ## (B, T, C)
        logits = self.lm_head(x)                 ## (B, T, vocab_sice)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B*T, C)
            targets  = targets.view(B*T)
            loss   = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        
        ## idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            
            ## crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            ## get the predictions
            logits, loss = self(idx_cond)
            ## focus only on last time stamp
            logits = logits[:, -1, :]           ## becomes (B, C)
            ## apply softmax to get probs
            probs = F.softmax(logits, dim= -1)    ## (B, C)
            ## sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)     ## (B, 1)
            ## append sample to the running sequence
            idx = torch.cat(  (idx, idx_next), dim=1  )            ## (B, T+1)
        return idx
            
            
            
######################################################################


model   = BigramLanguageModel()
m = model.to(device)

######################################################################

optimizer = torch.optim.Adam(  m.parameters(), lr=learning_rate   )

######################################################################

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    
    ## evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)   ## zero out
    loss.backward()
    optimizer.step()
    

################################################################
#### now, regenerate after some training


## Kick off generation with some starting token. In this case id 0

context = torch.zeros(  (1, 1),  dtype=torch.long, device=device   )

gen_text = m.generate(context, max_new_tokens=500)[0].tolist()

print(  decode(gen_text)   )









