import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from utils import generate_synthetic_data, idx_to_text

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        qkv = self.qkv_proj(x)  # (B, S, 3*E)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3 x B x heads x S x head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: B x heads x S x head_dim

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # B x heads x S x S
        attn_weights = F.softmax(attn_scores, dim=-1)  # B x heads x S x S

        attn_output = torch.matmul(attn_weights, v)  # B x heads x S x head_dim
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)  # B x S x E

        out = self.out_proj(attn_output)  # B x S x E
        return out, attn_weights

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, attn_weights = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, attn_weights

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        batch_size, seq_len = idx.size()
        x = self.token_emb(idx) + self.pos_emb[:, :seq_len]
        attn_weights_all = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            attn_weights_all.append(attn_weights)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, attn_weights_all

def train_loop(model, optimizer, criterion, data, epochs, device):
    model.train()
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for x, y in data:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            logits = logits.view(-1, logits.size(-1))
            y = y.view(-1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data)
        losses.append(avg_loss)
    return losses

def generate_text(model, start_seq, idx_to_token, token_to_idx, max_len, device):
    model.eval()
    generated = start_seq[:]
    input_ids = torch.tensor([token_to_idx[c] for c in start_seq], dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_len - len(start_seq)):
        logits, _ = model(input_ids)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        generated.append(idx_to_token[next_id])
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
    return ''.join(generated)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = list("abcdefghijklmnopqrstuvwxyz ")
    token_to_idx = {ch:i for i,ch in enumerate(vocab)}
    idx_to_token = {i:ch for ch,i in token_to_idx.items()}
    vocab_size = len(vocab)
    max_seq_len = 16

    # Generate synthetic dataset: sequences of random letters with next char as target
    dataset = generate_synthetic_data(token_to_idx, seq_len=max_seq_len, num_samples=5000)

    batch_size = 64
    data_batches = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        x = torch.stack([torch.tensor(d[0], dtype=torch.long) for d in batch])
        y = torch.stack([torch.tensor(d[1], dtype=torch.long) for d in batch])
        data_batches.append((x, y))

    embed_dim = 64
    num_heads = 4
    ff_dim = 256
    num_layers = 2

    model = TransformerLanguageModel(vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    losses = train_loop(model, optimizer, criterion, data_batches, epochs=10, device=device)

    # Check attention weights sum to 1
    model.eval()
    sample_x, _ = data_batches[0]
    sample_x = sample_x.to(device)
    with torch.no_grad():
        _, attn_weights_all = model(sample_x[:1])
    for attn_weights in attn_weights_all:
        sums = attn_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "Attention weights do not sum to 1"

    # Generate text starting from 'a'
    start_seq = ['a']
    generated_text = generate_text(model, start_seq, idx_to_token, token_to_idx, max_seq_len, device)

    print("Losses:", losses)
    print("Generated text:", generated_text)

if __name__ == "__main__":
    main()