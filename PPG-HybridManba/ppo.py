import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import HybridActorCritic


# ============================================================
# Pre-allocated Rollout Buffer (Policy Phase)
# ============================================================
class RolloutBuffer:
    """
    Fixed-size, pre-allocated tensor buffer for PPO rollout collection.
    Avoids Python list append/stack overhead entirely.
    """
    def __init__(self, buffer_size, state_dim, seq_len, device='cpu'):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0  # write pointer

        # Pre-allocate all tensors
        self.states = torch.zeros(buffer_size, state_dim, device=device)
        self.state_seqs = torch.zeros(buffer_size, seq_len, state_dim, device=device)
        self.actions = torch.zeros(buffer_size, dtype=torch.long, device=device)
        self.logprobs = torch.zeros(buffer_size, device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.is_terminals = torch.zeros(buffer_size, dtype=torch.bool, device=device)
        self.values = torch.zeros(buffer_size, device=device)  # V(s) from old policy
        self.action_probs_old = None  # Stored after policy phase for aux buffer

    def store(self, state, state_seq, action, logprob, value):
        """Store a single transition at current pointer (with overflow protection)."""
        if self.ptr >= self.buffer_size:
            return  # Buffer full — skip until update() clears it
        idx = self.ptr
        self.states[idx] = state
        self.state_seqs[idx] = state_seq
        self.actions[idx] = action
        self.logprobs[idx] = logprob
        self.values[idx] = value
        self.ptr += 1

    def store_reward_terminal(self, reward, is_terminal):
        """Store reward and terminal flag for the most recent transition."""
        idx = self.ptr - 1
        self.rewards[idx] = reward
        self.is_terminals[idx] = is_terminal

    @property
    def size(self):
        return self.ptr

    def clear(self):
        self.ptr = 0


# ============================================================
# Auxiliary Buffer (Auxiliary Phase - stores data from N_pi policy phases)
# ============================================================
class AuxiliaryBuffer:
    """
    Accumulates rollout data across multiple policy phases.
    Used by the PPG auxiliary phase to distill value knowledge.
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.state_seqs_list = []
        self.returns_list = []
        self.old_action_probs_list = []

    def store_from_policy_phase(self, state_seqs, returns, old_action_probs):
        """
        Archive data from a completed policy phase.
        
        Args:
            state_seqs: (N, seq_len, state_dim) tensor
            returns: (N,) tensor of GAE-computed returns
            old_action_probs: (N, action_dim) tensor of policy probs at collection time
        """
        self.state_seqs_list.append(state_seqs.detach())
        self.returns_list.append(returns.detach())
        self.old_action_probs_list.append(old_action_probs.detach())

    def get_all(self):
        """Concatenate all stored data into single tensors."""
        state_seqs = torch.cat(self.state_seqs_list, dim=0)
        returns = torch.cat(self.returns_list, dim=0)
        old_action_probs = torch.cat(self.old_action_probs_list, dim=0)
        return state_seqs, returns, old_action_probs

    @property
    def num_phases(self):
        return len(self.state_seqs_list)

    def clear(self):
        self.state_seqs_list.clear()
        self.returns_list.clear()
        self.old_action_probs_list.clear()


# ============================================================
# Phasic Policy Gradient (PPG) Agent
# ============================================================
class PPG:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 d_model=64, seq_len=16, d_state=16, d_conv=4, expand=2,
                 n_mamba_layers=3, n_heads=4, n_kv_heads=2,
                 # PPG-specific hyperparameters
                 gae_lambda=0.95,
                 mini_batch_size=64,
                 n_pi=4,          # Number of policy phases before one auxiliary phase
                 aux_epochs=6,    # Epochs for auxiliary phase
                 beta_clone=1.0,  # KL penalty weight in auxiliary phase
                 buffer_size=2048,
                 device='cpu'):
        """
        Phasic Policy Gradient agent with Hybrid Mamba Actor-Critic.
        
        Extends PPO with a two-phase training process:
          1. Policy Phase: Standard PPO update with GAE + mini-batches
          2. Auxiliary Phase: Distill value knowledge into shared backbone
             using KL-constrained joint optimization.

        Args:
            state_dim: Observation space dimension
            action_dim: Action space dimension (discrete)
            lr_actor: Learning rate for actor/shared parameters
            lr_critic: Learning rate for critic head
            gamma: Discount factor
            K_epochs: PPO epochs per policy phase
            eps_clip: PPO clipping parameter
            d_model: Model hidden dimension
            seq_len: Sequence context window length
            d_state: SSM state dimension
            d_conv: Convolution kernel size for Mamba
            expand: Mamba expansion factor
            n_mamba_layers: Number of Mamba blocks
            n_heads: Number of GQA query heads
            n_kv_heads: Number of GQA key/value heads
            gae_lambda: Lambda for Generalized Advantage Estimation
            mini_batch_size: Mini-batch size for gradient updates
            n_pi: Policy phases per auxiliary phase
            aux_epochs: Number of epochs in auxiliary phase
            beta_clone: KL divergence penalty weight
            buffer_size: Max transitions per policy phase
            device: Compute device
        """
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.seq_len = seq_len
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mini_batch_size = mini_batch_size
        self.n_pi = n_pi
        self.aux_epochs = aux_epochs
        self.beta_clone = beta_clone
        self.buffer_size = buffer_size

        # Rollout buffer (policy phase)
        self.buffer = RolloutBuffer(buffer_size, state_dim, seq_len, device)

        # Auxiliary buffer (accumulates across policy phases)
        self.aux_buffer = AuxiliaryBuffer(device)

        # Rolling state sequence tensor (avoids list manipulation in select_action)
        self._rolling_seq = torch.zeros(seq_len, state_dim, device=device)

        # Counter for policy phases
        self._policy_phase_count = 0

        # Initialize Hybrid Actor-Critic (with aux_critic head)
        self.policy = HybridActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            n_mamba_layers=n_mamba_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
        ).to(device)

        self.optimizer = optim.Adam([
            {'params': self.policy.encoder.parameters(), 'lr': lr_actor},
            {'params': self.policy.mamba.parameters(), 'lr': lr_actor},
            {'params': self.policy.attn.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic},
            {'params': self.policy.aux_critic.parameters(), 'lr': lr_critic},
        ])

        # Old policy for PPO clipping objective
        self.policy_old = HybridActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            n_mamba_layers=n_mamba_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    # ----------------------------------------------------------
    # Rolling sequence builder (replaces list-based approach)
    # ----------------------------------------------------------
    def _build_state_seq(self, state):
        """
        Build state sequence using a rolling tensor buffer.
        Shifts the tensor left by 1 and writes new state at the end.
        Much faster than list append + torch.stack.
        
        Args:
            state: Current state (numpy array or tensor)
        Returns:
            state_seq: (1, seq_len, state_dim) tensor
        """
        if isinstance(state, np.ndarray):
            state_t = torch.FloatTensor(state).to(self.device)
        else:
            state_t = state.to(self.device)

        # Roll left: shift all rows up by 1, write new state at bottom
        self._rolling_seq = torch.roll(self._rolling_seq, shifts=-1, dims=0)
        self._rolling_seq[-1] = state_t

        return self._rolling_seq.unsqueeze(0)  # (1, seq_len, state_dim)

    def select_action(self, state):
        """
        Select action using old policy with sequence context.
        Stores transition data directly into pre-allocated buffer tensors.
        """
        with torch.no_grad():
            state_seq = self._build_state_seq(state)
            action, action_logprob = self.policy_old.act(state_seq)
            # Also get value estimate from old policy for GAE
            _, state_value = self.policy_old(state_seq)

        state_t = torch.FloatTensor(state).to(self.device) if isinstance(state, np.ndarray) else state.to(self.device)

        self.buffer.store(
            state=state_t,
            state_seq=self._rolling_seq.clone(),  # (seq_len, state_dim)
            action=action.squeeze(),
            logprob=action_logprob.squeeze(),
            value=state_value.squeeze(),
        )

        return action.item()

    def reset_episode(self):
        """Reset the rolling sequence buffer (call on episode boundaries)."""
        self._rolling_seq.zero_()

    # ----------------------------------------------------------
    # GAE Computation
    # ----------------------------------------------------------
    def _compute_gae(self):
        """
        Compute Generalized Advantage Estimation (GAE) and returns.
        Called once before K_epochs of policy optimization.
        
        Uses V(s_last) as bootstrap value when buffer ends mid-episode,
        instead of incorrectly assuming 0.
        
        Returns:
            advantages: (N,) tensor
            returns: (N,) tensor
        """
        N = self.buffer.size
        advantages = torch.zeros(N, device=self.device)
        last_gae = 0.0

        # Bootstrap value: if last transition is NOT terminal,
        # estimate V(s_{N}) from old policy for proper GAE.
        with torch.no_grad():
            if not self.buffer.is_terminals[N - 1]:
                last_state_seq = self.buffer.state_seqs[N - 1].unsqueeze(0)
                _, bootstrap_val = self.policy_old(last_state_seq)
                bootstrap_value = bootstrap_val.squeeze().item()
            else:
                bootstrap_value = 0.0

        for t in reversed(range(N)):
            if t == N - 1:
                next_value = bootstrap_value
            else:
                next_value = self.buffer.values[t + 1].item()

            if self.buffer.is_terminals[t]:
                next_value = 0.0
                last_gae = 0.0

            delta = self.buffer.rewards[t] + self.gamma * next_value - self.buffer.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae

        returns = advantages + self.buffer.values[:N]
        return advantages, returns

    # ----------------------------------------------------------
    # Mini-batch generator
    # ----------------------------------------------------------
    def _mini_batch_generator(self, *tensors):
        """
        Yield shuffled mini-batches from the given tensors.
        All tensors must have the same first dimension.
        """
        N = tensors[0].size(0)
        indices = torch.randperm(N, device=self.device)

        for start in range(0, N, self.mini_batch_size):
            end = min(start + self.mini_batch_size, N)
            batch_idx = indices[start:end]
            yield tuple(t[batch_idx] for t in tensors)

    # ----------------------------------------------------------
    # Phase 1: Policy Phase Update
    # ----------------------------------------------------------
    def policy_phase_update(self):
        """
        PPG Policy Phase: Standard PPO update with GAE and mini-batches.
        
        After the update, archives data to the auxiliary buffer for
        later use in the auxiliary phase.
        
        Returns:
            dict with training metrics
        """
        N = self.buffer.size
        if N == 0:
            return {}

        # 1. Compute GAE advantages and returns ONCE (frozen)
        advantages, returns = self._compute_gae()

        # 2. Slice active buffer data
        old_state_seqs = self.buffer.state_seqs[:N].detach()
        old_actions = self.buffer.actions[:N].detach()
        old_logprobs = self.buffer.logprobs[:N].detach()

        # 3. Track metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        # 4. K epochs of mini-batch PPO
        for _ in range(self.K_epochs):
            for (mb_state_seqs, mb_actions, mb_logprobs,
                 mb_advantages, mb_returns) in self._mini_batch_generator(
                    old_state_seqs, old_actions, old_logprobs, advantages, returns):

                # Normalize advantages per mini-batch
                mb_adv = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Evaluate with current policy
                logprobs, state_values, dist_entropy, _ = self.policy.evaluate(mb_state_seqs, mb_actions)
                state_values = state_values.squeeze()

                # PPO ratio
                ratios = torch.exp(logprobs - mb_logprobs)

                # Clipped surrogate objective
                surr1 = ratios * mb_adv
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_adv

                # Loss: policy + value + entropy
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * self.MseLoss(state_values, mb_returns)
                entropy_loss = -0.01 * dist_entropy.mean()

                loss = policy_loss + value_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += dist_entropy.mean().item()
                n_updates += 1

        # 5. After policy update, snapshot action probs for auxiliary phase
        with torch.no_grad():
            _, _, _, old_action_probs = self.policy.evaluate(old_state_seqs, old_actions)

        # 6. Archive to auxiliary buffer
        self.aux_buffer.store_from_policy_phase(
            state_seqs=old_state_seqs,
            returns=returns,
            old_action_probs=old_action_probs,
        )

        # 7. Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 8. Clear rollout buffer
        self.buffer.clear()

        # 9. Increment phase counter
        self._policy_phase_count += 1

        metrics = {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
        }
        return metrics

    # ----------------------------------------------------------
    # Phase 2: Auxiliary Phase Update
    # ----------------------------------------------------------
    def auxiliary_phase_update(self):
        """
        PPG Auxiliary Phase: Distill value function knowledge into
        the shared backbone using a KL-constrained joint objective.
        
        Joint Loss = L_aux_value + L_main_value + beta_clone * L_kl
        
        Three objectives (per PPG paper):
            1. L_aux_value: MSE between aux_critic output and stored returns
               (trains auxiliary value head -> gradients flow into shared backbone)
            2. L_main_value: MSE between main critic output and stored returns
               (continues training critic with higher sample reuse)
            3. L_kl: KL divergence penalty to prevent policy drift
               (behavioral cloning constraint on actor)
        
        Returns:
            dict with auxiliary training metrics
        """
        if self.aux_buffer.num_phases == 0:
            return {}

        state_seqs, returns, old_action_probs = self.aux_buffer.get_all()

        total_aux_loss = 0.0
        total_main_value_loss = 0.0
        total_kl_loss = 0.0
        n_updates = 0

        for _ in range(self.aux_epochs):
            for mb_state_seqs, mb_returns, mb_old_probs in self._mini_batch_generator(
                    state_seqs, returns, old_action_probs):

                # Get current action probs, auxiliary value, AND main critic value
                current_probs, aux_values, main_values = self.policy.evaluate_aux(mb_state_seqs)
                aux_values = aux_values.squeeze()
                main_values = main_values.squeeze()

                # Objective 1: Auxiliary value loss (distills into shared backbone)
                aux_value_loss = 0.5 * self.MseLoss(aux_values, mb_returns)

                # Objective 2: Main critic loss (high sample reuse for value function)
                main_value_loss = 0.5 * self.MseLoss(main_values, mb_returns)

                # Objective 3: KL divergence penalty (prevent policy drift)
                # KL(old || new) = sum(old * log(old / new))
                kl_div = (mb_old_probs * (
                    torch.log(mb_old_probs + 1e-8) - torch.log(current_probs + 1e-8)
                )).sum(dim=-1).mean()

                # Joint loss: all 3 objectives per PPG paper
                loss = aux_value_loss + main_value_loss + self.beta_clone * kl_div

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_aux_loss += aux_value_loss.item()
                total_main_value_loss += main_value_loss.item()
                total_kl_loss += kl_div.item()
                n_updates += 1

        # Sync old policy after auxiliary phase
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear auxiliary buffer
        self.aux_buffer.clear()

        # Reset phase counter
        self._policy_phase_count = 0

        metrics = {
            'aux_value_loss': total_aux_loss / max(n_updates, 1),
            'main_value_loss_aux': total_main_value_loss / max(n_updates, 1),
            'kl_divergence': total_kl_loss / max(n_updates, 1),
        }
        return metrics

    # ----------------------------------------------------------
    # Combined update (called from training loop)
    # ----------------------------------------------------------
    def update(self):
        """
        Combined PPG update logic:
          1. Always run policy_phase_update (standard PPO + GAE)
          2. Every N_pi policy phases, run auxiliary_phase_update
        
        Returns:
            dict with all training metrics
        """
        # Phase 1: Policy Phase (always runs)
        metrics = self.policy_phase_update()

        # Phase 2: Auxiliary Phase (runs every N_pi policy phases)
        if self._policy_phase_count >= self.n_pi:
            aux_metrics = self.auxiliary_phase_update()
            metrics.update(aux_metrics)
            metrics['aux_phase_triggered'] = True
        else:
            metrics['aux_phase_triggered'] = False

        return metrics
