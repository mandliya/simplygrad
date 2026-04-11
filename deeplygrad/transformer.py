"""
deeplygrad.transformer: Transformer building blocks.

Components:
1. RoPE: (Rotary Position Embeddings)
2. LayerNorm
3. CausalSelfAttention (multi-head with causal mask)
4. MLP (Feed-forward network)
5. TransformerBlock
"""

from deeplygrad.backend import xp, BACKEND_NAME
from deeplygrad.tensor import Tensor, cat
import deeplygrad.nn as nn
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    d_model: int = 768
    n_heads: int = 12
    d_mlp: int = 3072
    max_seq_len: int = 2048
    dropout: float = 0.1
    n_layers: int = 12
    n_vocab: int = 50257
    n_ctx: int = 2048
    eps: float = 1e-8
    weight_decay: float = 0.01
    pos_base: int = 10000

class Rope(nn.Module):
    """
    Rotary Position Embeddings (RoPE) [Su et al., 2021]

    Encodes position by rotating query/key vectors per-head.
    Each dimension pair (x_i, x_{i+d/2}) is rotated by angle
    theta_i = pos / base^(2i/d_head), so the q·k dot product
    depends only on relative position.

    Expects input of shape (B, T, n_heads, d_head).
    """
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = cfg.d_model // cfg.n_heads
        assert self.d_head % 2 == 0, f"d_head must be even for RoPE, got {self.d_head}"

        ixs = xp.arange(0, self.d_head, step=2, dtype=xp.float64)       # (d_head/2,)
        freqs = 1.0 / (cfg.pos_base ** (ixs / self.d_head))             # (d_head/2,)

        positions = xp.arange(0, cfg.max_seq_len, dtype=xp.float64).reshape(-1, 1)  # (max_seq_len, 1)
        angles = positions * freqs.reshape(1, -1)                        # (max_seq_len, d_head/2)

        self.register_buffer('sin_cache', Tensor(xp.sin(angles), requires_grad=False))
        self.register_buffer('cos_cache', Tensor(xp.cos(angles), requires_grad=False))

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, n_heads, d_head) -> same shape with RoPE applied."""
        B, T, n_heads, d_head = x.shape
        d_half = d_head // 2

        # (T, d_half) -> (1, T, 1, d_half) for broadcasting
        cos_t = Tensor(self.cos_cache.data[:T].reshape(1, T, 1, d_half))
        sin_t = Tensor(self.sin_cache.data[:T].reshape(1, T, 1, d_half))

        x1 = x[:, :, :, :d_half]
        x2 = x[:, :, :, d_half:]

        out1 = x1 * cos_t - x2 * sin_t
        out2 = x1 * sin_t + x2 * cos_t

        return cat([out1, out2], axis=-1)
    

class Attention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = cfg.d_model // cfg.n_heads
        self.W_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = Rope(cfg)
        self.softmax = nn.Softmax(axis=-1)
        self.register_buffer('mask', Tensor(xp.tril(
            xp.ones((cfg.n_ctx, cfg.n_ctx))).reshape(1, 1, cfg.n_ctx, cfg.n_ctx), requires_grad=False))

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape

        q = self.W_q(x).reshape(B, T, self.n_heads, self.d_head)
        k = self.W_k(x).reshape(B, T, self.n_heads, self.d_head)
        v = self.W_v(x).reshape(B, T, self.n_heads, self.d_head)

        q = self.rope(q)
        k = self.rope(k)

        q = q.transpose(0, 2, 1, 3)                            # (B, n_heads, T, d_head)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scale = 1.0 / xp.sqrt(float(self.d_head))
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale          # (B, n_heads, T, T)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = self.softmax(attn)
        out = attn @ v                                          # (B, n_heads, T, d_head)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)       # (B, T, D)
        return self.W_o(out)

class MLP(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.d_mlp = cfg.d_mlp
        self.W_1 = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
        self.W_2 = nn.Linear(cfg.d_mlp, cfg.d_model, bias=False)
        self.gelu = nn.GELU()
    
    def forward(self, x: Tensor) -> Tensor:
        return self.W_2(self.gelu(self.W_1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.ln_2 = nn.LayerNorm(cfg.d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.token_embedding = nn.Embedding(cfg.n_vocab, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.lm_head = nn.Linear(cfg.d_model, cfg.n_vocab, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data = xp.random.randn(*module.weight.shape) * 0.02
            if module.bias is not None:
                module.bias.data = xp.zeros_like(module.bias.data)
        elif isinstance(module, nn.Embedding):
            module.W.data = xp.random.randn(*module.W.shape) * 0.02
        elif isinstance(module, nn.LayerNorm):
            module.gamma.data = xp.ones_like(module.gamma.data)
            module.beta.data = xp.zeros_like(module.beta.data)
    
    def forward(self, x: Tensor, targets: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        B, T = x.shape
        x = self.token_embedding(x)                              # (B, T, D)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                 # (B, T, n_vocab)
        if targets is not None:
            V = self.cfg.n_vocab
            loss = nn.CrossEntropyLoss()(
                logits.reshape(B * T, V), targets.reshape(B * T)
            )
            return logits, loss
        return logits, None

    def generate(self, x: Tensor, max_new_tokens: int) -> Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(x)
            logits = logits[:, -1, :]
            probs = nn.Softmax(axis=-1)(logits)
            next_token = Tensor(xp.argmax(probs.data, axis=-1).reshape(-1, 1))
            x = cat([x, next_token], axis=1)
        return x

if __name__ == "__main__":
    cfg = TransformerConfig()
    model = Transformer(cfg)
    x = Tensor(xp.random.randint(0, cfg.n_vocab, (1, 10)))
    targets = Tensor(xp.random.randint(0, cfg.n_vocab, (1, 10)))
    logits, loss = model(x, targets)
    print(loss)
    print(logits.shape)