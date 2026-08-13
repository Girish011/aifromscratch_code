"""MiniGPT model pieces shared by week2 notebooks.

Extracted from 7_mini_gpt_from_scratch.ipynb so other notebooks can:
    from mini_gpt import MiniGPT
(You cannot `import` a .ipynb directly — it is not a Python module.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a lower-triangular (causal) mask."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads   # split channels across heads
        self.scale = self.head_dim ** -0.5       # 1/sqrt(d_k) — same reason as nb6
        # One big Linear makes Q, K, V together (3 * embed_dim), then we split
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)  # mix heads back together

    def forward(self, x):
        # x: (B, T, C) = batch, time/seats, channels (embed_dim)
        B, T, C = x.shape

        # Project to Q,K,V then reshape into heads:
        #   (B, T, 3*C) → (B, T, 3, nh, hs) → (3, B, nh, T, hs)
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, num_heads, T, head_dim)

        # Scores: how much each query seat matches each key seat
        att = (q @ k.transpose(-2, -1)) * self.scale   # (B, nh, T, T)

        # Causal mask: keep lower triangle (tril), wipe upper with -inf
        #   before softmax, -inf → weight ≈ 0 after softmax
        mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T).to(x.device)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)                   # each query row sums to 1

        # Mix values with those weights, then merge heads → (B, T, C)
        y = att @ v
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    """One GPT block: Pre-LN causal attention + Pre-LN MLP, both with residuals."""

    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        # Position-wise MLP: expand → GELU → shrink (same idea as encoder FF)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        # Residual sidewalk + attention edit (after LayerNorm)
        x = x + self.attn(self.ln1(x))
        # Residual sidewalk + MLP edit (after LayerNorm)
        x = x + self.ff(self.ln2(x))
        return x
# Easy story: LN steadies the volume → attn mixes past seats → add back original;
#             LN again → MLP rewrites features per seat → add back again.


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, ff_dim=128, num_layers=3, block_size=32):
        super().__init__()
        # token_embed: each vocab id → a learnable vector of length embed_dim
        #   idx 8 ("B") → some 64-dim arrow meaning "B"
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # pos_embed: WHERE in the window (seat 0, 1, ..., block_size-1)
        #   GPT needs position because attention alone has no left/right order.
        #   Learned table (not sin/cos here) — one vector per seat index.
        self.pos_embed = nn.Parameter(torch.zeros(1, block_size, embed_dim))

        # Stack of GPT blocks (causal attn + MLP), run in sequence
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)          # final norm before the head
        self.head = nn.Linear(embed_dim, vocab_size) # hidden → vocab scores (logits)
        self.block_size = block_size
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Small random start (GPT-2-ish): helps stable early training
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx):
        # idx: (B, T) integer character ids, T must be ≤ block_size
        B, T = idx.shape
        assert T <= self.block_size

        tok_emb = self.token_embed(idx)       # (B, T, C) — WHAT the char is
        pos_emb = self.pos_embed[:, :T, :]    # (1, T, C) — WHERE it sits (broadcast over B)
        x = tok_emb + pos_emb                # combine "what" + "where"

        x = self.blocks(x)                   # deep causal transformer stack
        x = self.ln_f(x)
        logits = self.head(x)                # (B, T, vocab_size)
        return logits
        # Training: compare logits to y (next chars) with cross-entropy.

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Autoregressive sampling: append one char at a time."""
        # idx starts as a prompt, e.g. encode("Fir") → shape (1, 3)
        for _ in range(max_new_tokens):
            # Model can only see block_size seats — keep the most recent window
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)                 # (1, T, vocab)
            # Only the LAST seat predicts the brand-new next character
            logits = logits[:, -1, :] / temperature  # temperature: <1 sharper, >1 wilder
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # sample 1 id from probs
            idx = torch.cat([idx, next_id], dim=1)             # append → longer sequence
        return idx
# Easy story: forward = "score next char at every seat."
#             generate = "take last seat's scores → sample a char → glue it on → repeat."
