import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import pandas as pd
from models.policy_network import TradingPolicyNetwork
from experts.trend_following import TrendFollowingExpert
from utils.metrics import calculate_trading_metrics



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
    Imitation learning from the Trend‑Following expert.

    The expert produces signals in {‑1 (SELL), 0 (HOLD), +1 (BUY)}.
    The environment’s action space is exactly these three values.
    The policy network learns to output the same three signals (encoded
    as class indices 0, 1, 2) via supervised cloning and optional DAgger.
    """

    # Mapping from expert signal to class index for network training
    SIGNAL_TO_CLASS = {-1: 0, 0: 1, 1: 2}
    CLASS_TO_SIGNAL = {0: -1, 1: 0, 2: 1}

    def __init__(self, env, config: Dict):
        self.env = env
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Environment observation dimension
        self.input_dim = env.observation_space.shape[0]
        self.hidden_dims = config.get('hidden_dims', [256, 128, 64])
        # 3 output classes: SELL, HOLD, BUY
        self.output_dim = 3

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

        # Policy network – outputs 3 classes (expert signals)
        self.model = TradingPolicyNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            use_attention=config.get('use_attention', True)
        ).to(self.device)

        self.load_model("models/model_weights/imitation_policy.pth")


    # ------------------------------------------------------------------
    #  Expert demonstration collection
    # ------------------------------------------------------------------
    def collect_expert_demonstrations(self, num_episodes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the expert policy in the environment and record
        (observation, signal_class) pairs.
        """
        all_obs = []
        all_labels = []

        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            step = 0
            while not done and step < self.dagger_steps_per_episode:
                # 1. Expert raw signal (-1, 0, +1)
                signal = self.expert.get_expert_action(self.env.data, self.env.current_step)

                # 2. Record observation and the corresponding class label
                label = self.SIGNAL_TO_CLASS[signal]
                all_obs.append(obs.copy())
                all_labels.append(label)

                # 3. Step the environment directly with the expert's signal
                obs, reward, terminated, truncated, info = self.env.step(signal)
                done = terminated or truncated
                step += 1

            logger.info(f"Episode {ep+1}: collected {step} transitions")

        return np.array(all_obs), np.array(all_labels)

    # ------------------------------------------------------------------
    #  Behavioural cloning training
    # ------------------------------------------------------------------
    def train(self, num_expert_episodes: int = 10) -> nn.Module:
        """
        Collect expert demonstrations and train the policy via behavioural cloning.
        Optionally run DAgger iterations afterwards.
        """
        logger.info("Collecting expert demonstrations...")
        X, y = self.collect_expert_demonstrations(num_expert_episodes)

        split = int(len(X) * (1 - self.validation_split))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        self.model = self._behavioral_cloning(X_train, y_train, X_val, y_val)

        if self.use_dagger:
            self.model = self._dagger_training()

        return self.model

    def _behavioral_cloning(self, X_train, y_train, X_val, y_val):
        """Standard supervised training on signal classes."""
        train_dataset = ImitationDataset(X_train, y_train)
        val_dataset   = ImitationDataset(X_val, y_val)
        train_loader  = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader    = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        best_val_acc = 0
        best_state = None

        for epoch in range(self.epochs):
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

                if (batch_idx + 1) % 10 == 0:
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
            val_acc   = 100 * val_correct / val_total
            scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{self.epochs} | "
                  f"Train Loss: {train_loss/len(train_loader):.4f} | "
                  f"Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss/len(val_loader):.4f} | "
                  f"Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = self.model.state_dict().copy()

        self.model.load_state_dict(best_state)
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        return self.model

    # ------------------------------------------------------------------
    #  DAgger implementation
    # ------------------------------------------------------------------
    def _dagger_training(self) -> nn.Module:
        """DAgger: iterative dataset aggregation and retraining."""
        aggregated_obs = []
        aggregated_labels = []

        # Start with fresh expert demonstrations
        X_init, y_init = self.collect_expert_demonstrations(num_episodes=5)
        aggregated_obs.extend(X_init)
        aggregated_labels.extend(y_init)

        for it in range(self.dagger_iters):
            logger.info(f"DAgger iteration {it+1}/{self.dagger_iters}")

            new_obs, expert_labels, policy_labels = self._collect_policy_and_expert_actions()

            # Add samples where policy disagreed with expert
            for obs, pol_lbl, exp_lbl in zip(new_obs, policy_labels, expert_labels):
                if pol_lbl != exp_lbl:
                    aggregated_obs.append(obs)
                    aggregated_labels.append(exp_lbl)

            X_agg = np.array(aggregated_obs)
            y_agg = np.array(aggregated_labels)
            split = int(len(X_agg) * (1 - self.validation_split))
            self._behavioral_cloning(X_agg[:split], y_agg[:split],
                                     X_agg[split:], y_agg[split:])

        return self.model

    def _collect_policy_and_expert_actions(self) -> Tuple[List[np.ndarray], List[int], List[int]]:
        """
        Run current policy alongside expert.
        Returns:
            observations: list of observation arrays
            expert_labels: class indices (0,1,2) from expert
            policy_labels: class indices from current policy
        """
        observations = []
        expert_labels = []
        policy_labels = []

        for _ in range(5):
            obs, _ = self.env.reset()
            done = False
            step = 0
            while not done and step < self.dagger_steps_per_episode:
                # 1. Policy prediction (signal class)
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits, _, _ = self.model(obs_tensor)
                    pol_class = torch.argmax(logits, dim=1).item()
                pol_signal = self.CLASS_TO_SIGNAL[pol_class]

                # 2. Expert signal and its class
                exp_signal = self.expert.get_expert_action(self.env.data, self.env.current_step)
                exp_class = self.SIGNAL_TO_CLASS[exp_signal]

                # 3. Record observations and labels
                observations.append(obs.copy())
                expert_labels.append(exp_class)
                policy_labels.append(pol_class)

                # 4. Step environment using the POLICY action (as direct signal)
                obs, reward, terminated, truncated, info = self.env.step(pol_signal)
                done = terminated or truncated
                step += 1

        return observations, expert_labels, policy_labels

    # ------------------------------------------------------------------
    #  Model persistence
    # ------------------------------------------------------------------
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

        trading_metrics=[]
        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            
            done = False
            episode_reward = 0
            #print(DBG)
            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits, _, _ = self.model(obs_tensor)
                    pol_class = torch.argmax(logits, dim=1).item()
                signal = self.CLASS_TO_SIGNAL[pol_class]
                obs, reward, terminated, truncated, info = self.env.step(signal)
                episode_reward += reward
                done = terminated or truncated
                #print(f"Done: {done},ep:{ep} obs:{obs}, reward:{reward}")

            total_rewards.append(episode_reward)
            total_pnls.append(info['total_pnl'])
            trading_metrics.append(calculate_trading_metrics(info['trade_history']))

        df_trading_merics = pd.DataFrame(trading_metrics)

        return df_trading_merics


if __name__ == "__main__":
    from environments.trading_env import TradingEnvironment
    import yaml

    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    env = TradingEnvironment()

    imitation_cfg = {
        **cfg['imitation'],
        'fast_ma': 20,
        'slow_ma': 50,
        'use_dagger': False,          # Set to True for DAgger
        'dagger_iterations': 3,
        'dagger_rollout_steps': 500,
        'hidden_dims': [256, 128, 64],
        'use_attention': True,
    }

    learner = ImitationLearner(env, imitation_cfg)
    learner.train(num_expert_episodes=1)
    learner.save_model("models/model_weights/imitation_policy.pth")
    eval_results_df = learner.evaluate_in_env(num_episodes=1)
    print(f"{eval_results_df}")