import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

from models.policy_network import TradingPolicyNetwork
from experts.trend_following import TrendFollowingExpert
from utils.metrics import calculate_imitation_metrics

logger = logging.getLogger(__name__)

class ImitationDataset(Dataset):
    def __init__(self, observations: np.ndarray, actions: np.ndarray):
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.LongTensor(actions)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


class ImitationLearner:
    """
    Imitation learning from expert demonstrations.
    Uses the TradingEnvironment to generate (state, expert_action) pairs.
    """

    def __init__(self, env, config: Dict):
        self.env = env
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model architecture – input dimension from environment
        self.input_dim = env.observation_space.shape[0]
        self.hidden_dims = config.get('hidden_dims', [256, 128, 64])
        self.output_dim = 4   # BUY, SELL, HOLD, CLOSE

        # Training hyperparameters
        self.epochs = config.get('epochs', 50)
        self.batch_size = config.get('batch_size', 64)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.validation_split = config.get('validation_split', 0.2)

        # DAgger options
        self.use_dagger = config.get('use_dagger', False)
        self.dagger_iters = config.get('dagger_iterations', 5)
        self.dagger_steps_per_episode = config.get('dagger_rollout_steps', 500)

        # Expert (EMA crossover)
        self.expert = TrendFollowingExpert(
            fast_ma=config.get('fast_ma', 20),
            slow_ma=config.get('slow_ma', 50)
        )

        # Policy network
        self.model = TradingPolicyNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            use_attention=config.get('use_attention', True)
        ).to(self.device)

    def collect_expert_demonstrations(self, num_episodes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the expert policy in the environment and record (observation, expert_action).
        """
        all_obs = []
        all_actions = []

        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            self.expert.reset_position()
            done = False
            step = 0
            while not done and step < self.dagger_steps_per_episode:
                # Get expert action using the environment's current data and step index
                expert_action = self.expert.get_expert_action(self.env.data, self.env.current_step)
                all_obs.append(obs.copy())
                all_actions.append(expert_action)

                # Step environment using the expert's action
                obs, reward, terminated, truncated, info = self.env.step(expert_action)
                done = terminated or truncated
                step += 1

            logger.info(f"Episode {ep+1}: collected {step} transitions")

        return np.array(all_obs), np.array(all_actions)

    def train(self, num_expert_episodes: int = 10) -> nn.Module:
        """
        Collect expert demonstrations and train policy via behavioural cloning.
        Optionally then run DAgger iterations.
        """
        # 1. Collect expert data
        logger.info("Collecting expert demonstrations...")
        X, y = self.collect_expert_demonstrations(num_expert_episodes)

        # 2. Train/validation split
        split = int(len(X) * (1 - self.validation_split))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # 3. Behavioural cloning
        self.model = self._behavioral_cloning(X_train, y_train, X_val, y_val)

        # 4. DAgger iterative improvement (if enabled)
        if self.use_dagger:
            self.model = self._dagger_training()

        return self.model

    def _behavioral_cloning(self, X_train, y_train, X_val, y_val):
        """Standard supervised training."""
        train_dataset = ImitationDataset(X_train, y_train)
        val_dataset = ImitationDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        best_val_acc = 0
        best_state = None
        #print(DBG)

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            for batch_idx, (obs, act) in enumerate(train_loader):
                obs, act = obs.to(self.device), act.to(self.device)
                optimizer.zero_grad()
                logits, _, _ = self.model(obs)
                loss = criterion(logits, act)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pred = torch.argmax(logits, dim=1)
                train_correct += (pred == act).sum().item()
                train_total += act.size(0)
                
                if(batch_idx + 1) % 4 == 0:
                    # Print loss for this batch
                    print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
            # Validation
            self.model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for obs, act in val_loader:
                    obs, act = obs.to(self.device), act.to(self.device)
                    logits, _, _ = self.model(obs)
                    loss = criterion(logits, act)
                    val_loss += loss.item()
                    pred = torch.argmax(logits, dim=1)
                    val_correct += (pred == act).sum().item()
                    val_total += act.size(0)

            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            scheduler.step(val_loss)

            if (epoch + 1) % 1 == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"Train Loss: {train_loss/len(train_loader):.4f} | "
                    f"Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss/len(val_loader):.4f} | "
                    f"Val Acc: {val_acc:.2f}%"
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = self.model.state_dict().copy()

        self.model.load_state_dict(best_state)
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        return self.model

    def _dagger_training(self) -> nn.Module:
        """
        DAgger: collect new trajectories using current policy,
        query expert for correct actions, aggregate dataset, retrain.
        """
        aggregated_obs = []
        aggregated_actions = []

        # Initial dataset from expert demonstrations (can reuse previously collected)
        X_init, y_init = self.collect_expert_demonstrations(num_episodes=5)
        aggregated_obs.extend(X_init)
        aggregated_actions.extend(y_init)

        for it in range(self.dagger_iters):
            logger.info(f"DAgger iteration {it+1}/{self.dagger_iters}")

            # Collect policy rollouts and expert actions simultaneously
            new_obs, expert_actions, policy_actions = self._collect_policy_and_expert_actions()

            # Add only where policy action differs from expert action
            for obs, policy_act, expert_act in zip(new_obs, policy_actions, expert_actions):
                if policy_act != expert_act:
                    aggregated_obs.append(obs)
                    aggregated_actions.append(expert_act)

            # Retrain on aggregated dataset
            X_agg = np.array(aggregated_obs)
            y_agg = np.array(aggregated_actions)
            split = int(len(X_agg) * (1 - self.validation_split))
            self._behavioral_cloning(X_agg[:split], y_agg[:split], X_agg[split:], y_agg[split:])

        return self.model

    def _collect_policy_and_expert_actions(self) -> Tuple[List[np.ndarray], List[int], List[int]]:
        """
        Run current policy in environment, while also running expert side‑by‑side.
        Returns: observations, expert_actions, policy_actions.
        """
        observations = []
        expert_actions = []
        policy_actions = []

        for _ in range(5):  # collect 5 episodes per iteration
            obs, _ = self.env.reset()
            self.expert.reset_position()
            done = False
            step = 0
            while not done and step < self.dagger_steps_per_episode:
                # Policy action
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits, _, _ = self.model(obs_tensor)
                    policy_action = torch.argmax(logits, dim=1).item()

                # Expert action (using same environment state)
                expert_action = self.expert.get_expert_action(self.env.data, self.env.current_step)

                observations.append(obs.copy())
                expert_actions.append(expert_action)
                policy_actions.append(policy_action)

                # Step environment using the policy action (makes the next state)
                obs, reward, terminated, truncated, info = self.env.step(policy_action)
                done = terminated or truncated
                step += 1

        return observations, expert_actions, policy_actions

    def save_model(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        logger.info(f"Model loaded from {path}")

    def evaluate_in_env(self, num_episodes: int = 5) -> Dict:
        """Evaluate the learned policy in the environment."""
        total_rewards = []
        total_pnls = []
        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits, _, _ = self.model(obs_tensor)
                    action = torch.argmax(logits, dim=1).item()
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            total_rewards.append(episode_reward)
            total_pnls.append(info['total_pnl'])
        return {'avg_reward': np.mean(total_rewards), 'avg_pnl': np.mean(total_pnls)}


if __name__ == "__main__":
    from environments.trading_env import TradingEnvironment
    import yaml

    # Load configurations
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Create environment (it loads its own data from data_config.yaml)
    env = TradingEnvironment()

    # Imitation learning config
    imitation_cfg = {
        **cfg['imitation'],
        'fast_ma': 20,
        'slow_ma': 50,
        'use_dagger': False,          # Set to True to enable DAgger
        'dagger_iterations': 3,
        'dagger_rollout_steps': 500,
        'hidden_dims': [256, 128, 64],
        'use_attention': True,
    }

    learner = ImitationLearner(env, imitation_cfg)

    # Train
    learner.train(num_expert_episodes=1)

    # Save model
    learner.save_model("models/model_weights/imitation_policy.pth")

    # Evaluate
    eval_results = learner.evaluate_in_env(num_episodes=5)
    print(f"Evaluation results: {eval_results}")