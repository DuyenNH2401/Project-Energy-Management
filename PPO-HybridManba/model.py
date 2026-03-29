import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import math

# ============================================================
# Mamba Block (Selective State Space Model - Pure PyTorch)
# ============================================================
class MambaBlock(nn.Module):
    """
    A single Mamba block implementing the Selective State Space Model.
    
    Architecture:
        input -> Linear expand -> Conv1D -> SSM -> Gated output -> Linear project
    
    Args:
        d_model: Input/output dimension
        d_state: SSM state expansion factor (N)
        d_conv: Local convolution width
        expand: Block expansion factor (E)
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)

        # Input projection: project to 2 * d_inner (for x and gate z)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise convolution over the sequence
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # SSM parameters - input-dependent (selective)
        # x -> (dt, B, C) projections
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # dt_rank=1 simplified

        # dt projection (from rank-1 to d_inner)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # A parameter - initialized with HiPPO-like structure
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))  # log for numerical stability
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Input projection -> (batch, seq_len, 2 * d_inner)
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)  # each (batch, seq_len, d_inner)

        # Conv1D over sequence dimension
        x_conv = x_proj.transpose(1, 2)  # (batch, d_inner, seq_len)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # trim to original length
        x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, d_inner)
        x_conv = F.silu(x_conv)

        # SSM parameters from input (selective mechanism)
        x_ssm = self.x_proj(x_conv)  # (batch, seq_len, d_state*2 + 1)
        dt, B, C = x_ssm.split([1, self.d_state, self.d_state], dim=-1)

        # dt projection and softplus
        dt = self.dt_proj(dt)  # (batch, seq_len, d_inner)
        dt = F.softplus(dt)

        # Discretize A
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # SSM scan (sequential for simplicity)
        y = self._ssm_scan(x_conv, dt, A, B, C)

        # Skip connection with D
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv

        # Gated output
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)
        return output

    def _ssm_scan(self, x, dt, A, B, C):
        """
        Selective SSM scan (sequential implementation).
        
        Args:
            x: (batch, seq_len, d_inner) - input after conv
            dt: (batch, seq_len, d_inner) - discretization step
            A: (d_inner, d_state) - state matrix
            B: (batch, seq_len, d_state) - input-dependent B
            C: (batch, seq_len, d_state) - input-dependent C
        Returns:
            y: (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape
        d_state = self.d_state

        # Initialize hidden state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        ys = []
        for t in range(seq_len):
            # Discretize: dA = exp(dt * A), dB = dt * B
            dt_t = dt[:, t, :].unsqueeze(-1)      # (batch, d_inner, 1)
            dA = torch.exp(dt_t * A.unsqueeze(0))  # (batch, d_inner, d_state)
            dB = dt_t * B[:, t, :].unsqueeze(1)    # (batch, d_inner, d_state) via broadcast

            # SSM recurrence: h = dA * h + dB * x
            x_t = x[:, t, :].unsqueeze(-1)         # (batch, d_inner, 1)
            h = dA * h + dB * x_t

            # Output: y = C^T * h
            C_t = C[:, t, :].unsqueeze(1)           # (batch, 1, d_state)
            y_t = (h * C_t).sum(dim=-1)             # (batch, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (batch, seq_len, d_inner)
        return y


class MambaBlockStack(nn.Module):
    """Stack of Mamba blocks with residual connections and LayerNorm."""
    def __init__(self, d_model, n_layers=3, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x: (batch, seq_len, d_model)
        """
        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x))  # Pre-norm residual
        return x


# ============================================================
# Grouped Query Attention (GQA)
# ============================================================
class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).
    
    Uses fewer key/value heads than query heads for efficiency.
    
    Args:
        d_model: Model dimension
        n_heads: Number of query heads
        n_kv_heads: Number of key/value heads (must divide n_heads)
    """
    def __init__(self, d_model, n_heads=4, n_kv_heads=2):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # repetition factor
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm = nn.LayerNorm(d_model)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)

        batch, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat K and V to match Q heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        output = self.out_proj(attn_output)
        return residual + output  # Residual connection


# ============================================================
# Hybrid Actor-Critic (Encoder -> Mamba x3 -> GQA -> Heads)
# ============================================================
class HybridActorCritic(nn.Module):
    """
    Hybrid Actor-Critic using Mamba + GQA.
    
    Architecture:
        Encoder (MLP) -> Mamba x 3 -> GQA x 1 -> Shared Representation
            ├── Actor Head (policy)
            └── Critic Head (value)
    
    Args:
        state_dim: Observation dimension
        action_dim: Number of discrete actions
        d_model: Hidden dimension (must be divisible by n_heads)
        d_state: SSM state dimension for Mamba
        d_conv: Convolution kernel size for Mamba
        expand: Expansion factor for Mamba
        n_mamba_layers: Number of stacked Mamba blocks
        n_heads: Number of attention query heads
        n_kv_heads: Number of attention key/value heads
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
        n_mamba_layers=3,
        n_heads=4,
        n_kv_heads=2,
    ):
        super().__init__()

        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
        )

        # --- Mamba Stack (temporal modeling) ---
        self.mamba = MambaBlockStack(
            d_model=d_model,
            n_layers=n_mamba_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # --- GQA (spatial focus) ---
        self.attn = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
        )

        # --- Final LayerNorm ---
        self.final_norm = nn.LayerNorm(d_model)

        # --- Actor Head ---
        self.actor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, action_dim),
            nn.Softmax(dim=-1),
        )

        # --- Critic Head ---
        self.critic = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_seq):
        """
        Forward pass through the hybrid architecture.
        
        Args:
            x_seq: (batch, seq_len, state_dim) - sequence of observations
        Returns:
            action_probs: (batch, action_dim) - policy from last timestep
            state_value: (batch, 1) - value from last timestep
        """
        # Encode each observation
        x = self.encoder(x_seq)          # (batch, seq_len, d_model)

        # Temporal modeling with Mamba
        z = self.mamba(x)                 # (batch, seq_len, d_model)

        # Spatial focus with GQA (residual is built-in)
        h = self.attn(z)                  # (batch, seq_len, d_model)

        # Final norm
        h = self.final_norm(h)

        # Use last timestep representation for decision
        h_last = h[:, -1, :]              # (batch, d_model)

        action_probs = self.actor(h_last)
        state_value = self.critic(h_last)

        return action_probs, state_value

    def act(self, state_seq):
        """
        Select action from a state sequence.
        
        Args:
            state_seq: (1, seq_len, state_dim) or (seq_len, state_dim)
        Returns:
            action, action_logprob
        """
        if state_seq.dim() == 2:
            state_seq = state_seq.unsqueeze(0)  # add batch dim

        action_probs, _ = self.forward(state_seq)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state_seqs, actions):
        """
        Evaluate actions given state sequences.
        
        Args:
            state_seqs: (batch, seq_len, state_dim)
            actions: (batch,)
        Returns:
            action_logprobs, state_values, dist_entropy
        """
        action_probs, state_values = self.forward(state_seqs)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        return action_logprobs, state_values, dist_entropy
