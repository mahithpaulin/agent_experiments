import pytest
import torch
from main import MultiHeadSelfAttention, TransformerLanguageModel, train_loop, generate_text, generate_synthetic_data

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def vocab_and_mappings():
    vocab = list("abcdefghijklmnopqrstuvwxyz ")
    token_to_idx = {ch:i for i,ch in enumerate(vocab)}
    idx_to_token = {i:ch for ch,i in token_to_idx.items()}
    return vocab, token_to_idx, idx_to_token

def test_attention_weights_sum_to_one(device):
    embed_dim = 32
    num_heads = 4
    seq_len = 8
    batch_size = 2
    attn = MultiHeadSelfAttention(embed_dim, num_heads).to(device)
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)
    _, attn_weights = attn(x)
    sums = attn_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

def test_training_loss_decreases(device, vocab_and_mappings):
    vocab, token_to_idx, idx_to_token = vocab_and_mappings
    vocab_size = len(vocab)
    max_seq_len = 16
    dataset = generate_synthetic_data(token_to_idx, seq_len=max_seq_len, num_samples=200)
    batch_size = 32
    data_batches = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        x = torch.stack([torch.tensor(d[0], dtype=torch.long) for d in batch])
        y = torch.stack([torch.tensor(d[1], dtype=torch.long) for d in batch])
        data_batches.append((x, y))
    embed_dim = 32
    num_heads = 4
    ff_dim = 128
    num_layers = 1
    model = TransformerLanguageModel(vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    losses = train_loop(model, optimizer, criterion, data_batches, epochs=5, device=device)
    assert losses[0] > losses[-1], "Loss did not decrease over training"

def test_generate_text_coherence(device, vocab_and_mappings):
    vocab, token_to_idx, idx_to_token = vocab_and_mappings
    vocab_size = len(vocab)
    max_seq_len = 16
    dataset = generate_synthetic_data(token_to_idx, seq_len=max_seq_len, num_samples=500)
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
    criterion = torch.nn.CrossEntropyLoss()
    train_loop(model, optimizer, criterion, data_batches, epochs=10, device=device)
    start_seq = ['a']
    generated_text = generate_text(model, start_seq, idx_to_token, token_to_idx, max_seq_len, device)
    # Check generated text length and characters in vocab
    assert len(generated_text) == max_seq_len
    assert all(c in vocab for c in generated_text)

def test_attention_weights_all_layers_sum_to_one(device):
    vocab_size = 30
    embed_dim = 32
    num_heads = 4
    ff_dim = 128
    num_layers = 3
    max_seq_len = 10
    model = TransformerLanguageModel(vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_len).to(device)
    x = torch.randint(0, vocab_size, (2, max_seq_len), device=device)
    with torch.no_grad():
        _, attn_weights_all = model(x)
    for attn_weights in attn_weights_all:
        sums = attn_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

def test_generate_synthetic_data_shapes(vocab_and_mappings):
    _, token_to_idx, _ = vocab_and_mappings
    seq_len = 12
    num_samples = 50
    data = generate_synthetic_data(token_to_idx, seq_len=seq_len, num_samples=num_samples)
    assert len(data) == num_samples
    for inp, tgt in data:
        assert len(inp) == seq_len - 1
        assert len(tgt) == seq_len - 1
        assert all(isinstance(i, int) for i in inp)
        assert all(isinstance(i, int) for i in tgt)