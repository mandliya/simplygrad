"""
GPT-2 (character-level) using deeplygrad

Trains a small transformer language model on TinyShakespeare using only
the deeplygrad library — no PyTorch or other ML frameworks.

Components used:
  - Transformer / TransformerConfig  (deeplygrad.transformer)
  - Adam optimizer                   (deeplygrad.optim)
  - CharTokenizer                    (tokenizer)
  - Tensor / xp                      (deeplygrad core)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deeplygrad.transformer import Transformer, TransformerConfig
from deeplygrad.optim import Adam
from deeplygrad.backend import xp, BACKEND_NAME
from deeplygrad.tensor import Tensor
from tokenizer import CharTokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import requests

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
BLOCK_SIZE = 128
MAX_STEPS = 2000
EVAL_INTERVAL = 200
EVAL_ITERS = 20
LEARNING_RATE = 3e-4
GENERATE_EVERY = 500
GENERATE_TOKENS = 200

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_data():
    cache_path = os.path.join(os.path.dirname(__file__), "tinyshakespeare.txt")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return f.read()
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    response.raise_for_status()
    with open(cache_path, "w") as f:
        f.write(response.text)
    return response.text


def get_batch(split, train_data, val_data):
    """Sample a random batch of (input, target) pairs for language modeling."""
    data = train_data if split == "train" else val_data
    max_start = len(data) - BLOCK_SIZE - 1
    offsets = xp.random.randint(0, max_start, size=(BATCH_SIZE,))
    x = xp.stack([data[i : i + BLOCK_SIZE] for i in offsets])
    y = xp.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in offsets])
    return Tensor(x, requires_grad=False), Tensor(y, requires_grad=False)


def estimate_loss(model, train_data, val_data):
    """Average loss over EVAL_ITERS batches for both splits."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        total = 0.0
        for _ in range(EVAL_ITERS):
            xb, yb = get_batch(split, train_data, val_data)
            _, loss = model(xb, yb)
            total += loss.item()
        out[split] = total / EVAL_ITERS
    model.train()
    return out


def generate_sample(model, tokenizer, prompt="\n", max_tokens=GENERATE_TOKENS):
    """Generate text from the model and return as a string."""
    model.eval()
    ids = tokenizer.encode(prompt)
    context = Tensor(xp.array(ids, dtype=xp.int64).reshape(1, -1))
    output = model.generate(context, max_tokens)
    model.train()
    return tokenizer.decode(output.data[0].astype(int).tolist())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Using backend: {BACKEND_NAME}")
    xp.random.seed(42)

    # 1. Load and tokenize data
    print("Loading TinyShakespeare...")
    text = get_data()
    tokenizer = CharTokenizer()
    tokenizer.train(text)
    print(f"Corpus: {len(text):,} characters, vocab size: {tokenizer.vocab_size}")

    all_ids = xp.array(tokenizer.encode(text), dtype=xp.int64)
    split = int(0.9 * len(all_ids))
    train_data = all_ids[:split]
    val_data = all_ids[split:]
    print(f"Train tokens: {len(train_data):,}  |  Val tokens: {len(val_data):,}")

    # 2. Build model with a tiny config
    cfg = TransformerConfig(
        d_model=64,
        n_heads=4,
        d_mlp=256,
        n_layers=4,
        n_vocab=tokenizer.vocab_size,
        n_ctx=BLOCK_SIZE,
        max_seq_len=BLOCK_SIZE,
        dropout=0.1,
    )
    model = Transformer(cfg)
    n_params = sum(p.data.size for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training loop
    train_losses = []
    val_losses = []
    step_timestamps = []

    print(f"\nTraining for {MAX_STEPS} steps  (batch={BATCH_SIZE}, block={BLOCK_SIZE}, lr={LEARNING_RATE})")
    print("-" * 70)

    t0 = time.time()
    for step in range(MAX_STEPS):

        # Evaluate periodically
        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
            losses = estimate_loss(model, train_data, val_data)
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])
            step_timestamps.append(step)
            elapsed = time.time() - t0
            print(
                f"step {step:5d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | "
                f"elapsed {elapsed:.1f}s"
            )

        # Generate a sample occasionally
        if step > 0 and step % GENERATE_EVERY == 0:
            sample = generate_sample(model, tokenizer)
            print(f"\n--- sample at step {step} ---")
            print(sample[:300])
            print("---\n")

        # Training step
        xb, yb = get_batch("train", train_data, val_data)
        optimizer.zero_grad()
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time:.1f}s  ({total_time / MAX_STEPS:.3f}s/step)")

    # 4. Final generation
    print("\n" + "=" * 70)
    print("GENERATED TEXT")
    print("=" * 70)
    sample = generate_sample(model, tokenizer, prompt="\n", max_tokens=500)
    print(sample)

    # 5. Loss plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(step_timestamps, train_losses, "o-", label="Train")
    ax.plot(step_timestamps, val_losses, "s--", label="Validation")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("GPT-2 (char-level) Training on TinyShakespeare")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_path = os.path.join(os.path.dirname(__file__), "gpt2_loss.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"\nLoss plot saved to {plot_path}")


if __name__ == "__main__":
    main()
