import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
n_head = 4
n_layer = 3
dropout = 0.1
num_heads = 4 # Number of heads
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars) #65
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module): 

    """ one head of self-attention """

    def __init__(self, head_size): 
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query =  nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x): 
        B, T, C = x.shape # 32, 8, 32

        key = self.key(x) # B, T, H --> H: Headsize
        query = self.query(x) # B, T, H
        

        w = query @ key.transpose(-2, -1) * key.shape[-1]**-0.5 # (B,T, H) @ (B, H, T)  --> (B, T, T)
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(w, dim=-1) # Last Dimension 

        w = self.dropout(w)

        value = self.value(x) # B, T, H
        out = w @ value # (B, T, T) @ (B, T, H) = (B, T, H)

        return out

class MultiHeadAttention(nn.Module): 
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):  # number_of_head
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x): 
        x = torch.cat([h(x) for h in self.heads], dim=-1) # concat on C
        out = self.proj(x) # mix the heads before FFD
        out = self.dropout(out)
        return out # B, T, C

class FeedForwardNetwork(nn.Module): 
    """ a simple non-linear transformation after MultiheadAttention"""
    
    def __init__(self, n_embd): 
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), # 4 times input size as the gpt paper
            nn.ReLU(), # non-linearity
            nn.Linear(4*n_embd, n_embd), # reduce to input_size
            nn.Dropout(dropout)
        )

    def forward(self, x): 
        out =  self.ff(x)
        return out


class Block(nn.Module): 
    """ Transformer block: MulitheadAttention + FeedForwardNetwork """
    def __init__(self, n_embd, n_head): # n_head = 4
        super().__init__()
        head_size = n_embd // n_head # 32/4 = 8
        self.multi_head = MultiHeadAttention(n_head, head_size)  # B, T, C: init(4, 8) --> output: (B, T, 4*8)
        self.ffwd = FeedForwardNetwork(n_embd) 
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    
    def forward(self, x): 
        # Residual Network: f(x) = x + f(x)
        x = x + self.multi_head(self.ln1(x)) # B, T, C, PreNorm, norm before residual, so x still have direct path in backpropgation. 
        out = x + self.ffwd(self.ln2(x)) # B, T, C

        return out


    
        

class GPT(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # B, T, C
        self.token_pos_table = nn.Embedding(block_size, n_embd) # T, C  position table
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        
        self.lm_head = nn.Linear(n_embd, vocab_size) # B, T, vocab_size
        

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.token_pos_table(torch.arange(T, device=device)) #  T, C
        tok_pos_emb = tok_emb + pos_emb # B, T, C
        x = self.blocks(tok_pos_emb) # B, T, C
        logits = self.lm_head(x) # B, T, vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T) # B, T, Vocab_size --> B*T, Vocabsize --? (256, 8)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            last_idx = idx[:, -block_size:] # [T, -block_size] -> [1, -8]
            logits, loss = self(last_idx) 
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPT(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print("****With Dropout 10%*****")

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))