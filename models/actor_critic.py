import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    """Simple shared‑body actor‑critic network."""
    def __init__(self, input_dim, action_dim, hidden=[256, 128]):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden[1], action_dim)   # raw logits
        self.critic = nn.Linear(hidden[1], 1)            # state value

    def forward(self, x):
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

if __name__ == "__main__":
    # -------------------------------
    # 1. Set up device and parameters
    # -------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    input_dim = 100       # typical observation size
    action_dim = 3        # e.g., -1, 0, 1 encoded as 0,1,2
    batch_size = 32

    model = ActorCritic(input_dim, action_dim).to(device)
    print(model)

    # -------------------------------
    # 2. Generate dummy batch of observations
    # -------------------------------
    dummy_obs = torch.randn(batch_size, input_dim).to(device)

    # -------------------------------
    # 3. Forward pass
    # -------------------------------
    logits, values = model(dummy_obs)

    print(f"Input shape:  {dummy_obs.shape}")
    print(f"Logits shape: {logits.shape}   (expected: [{batch_size}, {action_dim}])")
    print(f"Values shape: {values.shape}   (expected: [{batch_size}, 1])")

    # -------------------------------
    # 4. Sample actions from the logits
    # -------------------------------
    dist = Categorical(logits=logits)
    actions = dist.sample()                     # tensor of shape [batch_size]
    log_probs = dist.log_prob(actions)          # tensor of shape [batch_size]

    print(f"Sampled actions: {actions}")
    print(f"Action log probs: {log_probs}")

    # -------------------------------
    # 5. Quick gradient check
    # -------------------------------
    # Combine actor and critic loss (just for testing)
    actor_loss = -log_probs.mean()              # negative log-likelihood
    critic_loss = nn.MSELoss()(values.squeeze(), torch.zeros_like(values.squeeze()))  # dummy target
    total_loss = actor_loss + 0.5 * critic_loss

    total_loss.backward()
    print(f"\nTotal loss: {total_loss.item():.4f}")
    print("Gradient check: no errors, backward pass successful.")