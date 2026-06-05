import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt

from environments.trading_env import TradingEnvironment
from environments.competitive_trading_env import CompetitiveTradingEnv
from models.actor_critic import ActorCritic
from utils.metrics import calculate_trading_metrics


class CompetitivePPOTrainer:
    def __init__(self, env: CompetitiveTradingEnv,
                 lr=3e-4, gamma=0.99, lam=0.95,
                 clip_eps=0.2, ppo_epochs=10, batch_size=64,
                 device='cpu', beta=0.1):
        """
        beta: competition coefficient.
              - 0.0 -> purely independent rewards (both can win)
              - >0   -> adds a small zero-sum term: agent gets bonus
                        for beating the opponent.
        """
        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.device = device
        self.beta = beta

        obs_dim = env.obs_dim
        action_dim = 3  # SELL(-1), HOLD(0), BUY(1) encoded as 0,1,2

        self.agent0 = ActorCritic(obs_dim, action_dim).to(device)
        self.agent1 = ActorCritic(obs_dim, action_dim).to(device)
        self.opt0 = optim.Adam(self.agent0.parameters(), lr=lr)
        self.opt1 = optim.Adam(self.agent1.parameters(), lr=lr)

        self.mem0 = {'s': [], 'a': [], 'logp': [], 'r': [], 'v': [], 'd': []}
        self.mem1 = {'s': [], 'a': [], 'logp': [], 'r': [], 'v': [], 'd': []}

    def act(self, obs, agent_net):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        logits, value = agent_net(obs_t)
        dist = Categorical(logits=logits)
        action_class = dist.sample()
        log_prob = dist.log_prob(action_class)
        return action_class.item(), log_prob.item(), value.item()

    def store(self, agent_id, s, a, logp, r, v, d):
        mem = self.mem0 if agent_id == 0 else self.mem1
        mem['s'].append(s)
        mem['a'].append(a)
        mem['logp'].append(logp)
        mem['r'].append(r)
        mem['v'].append(v)
        mem['d'].append(d)

    def collect_rollout(self, steps=2048):
        obs0, obs1 = self.env.reset()
        # optional: keep track of buy&hold price
        base_env = self.env.base_env
        initial_price = base_env.data.iloc[base_env.current_step]['close']
        for _ in range(steps):
            a0, logp0, val0 = self.act(obs0, self.agent0)
            a1, logp1, val1 = self.act(obs1, self.agent1)

            signal0 = a0 - 1
            signal1 = a1 - 1

            next_obs0, next_obs1, (rew0_ind, rew1_ind), done, info = self.env.step((signal0, signal1))

            # ---- compute buy&hold change ----
            current_price = base_env.data.iloc[base_env.current_step]['close']
            # price_change_ratio (or absolute change, adjust to your reward scale)
            bh_change = (current_price - initial_price) / initial_price  # return-like
            # You can convert to same units as PnL if needed
            initial_price = current_price  # update for next step (or keep original for cumulative?)

            # ---- reward = own_pnl_change - baseline + competition ----
            rew0 = (rew0_ind - bh_change) + self.beta * ((rew0_ind - bh_change) - (rew1_ind - bh_change))
            rew1 = (rew1_ind - bh_change) + self.beta * ((rew1_ind - bh_change) - (rew0_ind - bh_change))

            self.store(0, obs0, a0, logp0, rew0, val0, done)
            self.store(1, obs1, a1, logp1, rew1, val1, done)

            obs0, obs1 = next_obs0, next_obs1
            if done:
                obs0, obs1 = self.env.reset()
                # reset baseline price
                initial_price = base_env.data.iloc[base_env.current_step]['close']
        return obs0, obs1


    def compute_gae(self, mem, last_obs, agent_net):
        with torch.no_grad():
            obs_t = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
            _, last_val = agent_net(obs_t)
            last_val = last_val.item()
        values = mem['v'] + [last_val]
        dones = mem['d'] + [False]
        rewards = mem['r']
        n = len(rewards)
        returns = np.zeros(n, dtype=np.float32)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0
        for i in reversed(range(n)):
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            returns[i] = gae + values[i]
            advantages[i] = gae
        mem['ret'] = returns
        mem['adv'] = advantages

    def update_agent(self, agent_net, optimizer, mem):
        states = torch.FloatTensor(np.array(mem['s'])).to(self.device)
        actions = torch.LongTensor(mem['a']).to(self.device)
        old_logprobs = torch.FloatTensor(mem['logp']).to(self.device)
        returns = torch.FloatTensor(mem['ret']).to(self.device)
        advantages = torch.FloatTensor(mem['adv']).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(len(states))
            for start in range(0, len(states), self.batch_size):
                idx = indices[start:start + self.batch_size]
                batch_s = states[idx]
                batch_a = actions[idx]
                batch_old_logp = old_logprobs[idx]
                batch_adv = advantages[idx]
                batch_ret = returns[idx]

                logits, val_pred = agent_net(batch_s)
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(batch_a)
                entropy = dist.entropy().mean()

                ratio = (new_logp - batch_old_logp).exp()
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * (val_pred.squeeze() - batch_ret).pow(2).mean()
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent_net.parameters(), 0.5)
                optimizer.step()

    def clear_memory(self):
        self.mem0 = {'s': [], 'a': [], 'logp': [], 'r': [], 'v': [], 'd': []}
        self.mem1 = {'s': [], 'a': [], 'logp': [], 'r': [], 'v': [], 'd': []}

    def train(self, total_steps=100_000, rollout_steps=2048,
              save_path='models/competitive'):
        step = 0
        pnl_history = []
        while step < total_steps:
            last_obs0, last_obs1 = self.collect_rollout(rollout_steps)
            step += rollout_steps

            self.compute_gae(self.mem0, last_obs0, self.agent0)
            self.compute_gae(self.mem1, last_obs1, self.agent1)

            self.update_agent(self.agent0, self.opt0, self.mem0)
            self.update_agent(self.agent1, self.opt1, self.mem1)

            self.clear_memory()

            pnl0 = self.env.pnls[0]
            pnl1 = self.env.pnls[1]
            pnl_history.append((step, pnl0, pnl1))
            print(f"Step {step:6d} | PnL0: {pnl0:8.2f} | PnL1: {pnl1:8.2f}")

        torch.save(self.agent0.state_dict(), f"{save_path}_agent0.pth")
        torch.save(self.agent1.state_dict(), f"{save_path}_agent1.pth")
        print("Training finished. Models saved.")
        return pnl_history


def evaluate_and_plot(test_env: CompetitiveTradingEnv,
                      trainer: CompetitivePPOTrainer,
                      max_steps=500):
    test_env.reset()
    obs0, obs1 = test_env.reset()
    done = False
    step = 0

    episode_data = {
        'steps': [],
        'time': [],
        'prices': [],
        'open': [], 'high': [], 'low': [],
        'positions': {0: [], 1: []},
        'pnl': {0: [], 1: []},
    }

    while not done and step < max_steps:
        a0_class, _, _ = trainer.act(obs0, trainer.agent0)
        a1_class, _, _ = trainer.act(obs1, trainer.agent1)
        signal0 = a0_class - 1
        signal1 = a1_class - 1

        obs0, obs1, (r0, r1), done, info = test_env.step((signal0, signal1))

        base_env = test_env.base_env
        row = base_env.data.iloc[base_env.current_step]
        cur_price = row['close']
        episode_data['steps'].append(step)
        episode_data['time'].append(row['time'])
        episode_data['prices'].append(cur_price)
        episode_data['open'].append(row['open'])
        episode_data['high'].append(row['high'])
        episode_data['low'].append(row['low'])
        episode_data['positions'][0].append(test_env.positions[0])
        episode_data['positions'][1].append(test_env.positions[1])
        episode_data['pnl'][0].append(test_env.pnls[0])
        episode_data['pnl'][1].append(test_env.pnls[1])
        step += 1

    CompetitiveTradingEnv.visualize_episode(episode_data,
                                            title="Competitive Agents Test")

    print("\n=== Trading Metrics ===")
    for agent_id in [0, 1]:
        trades = test_env.trade_histories[agent_id]
        if trades:
            metrics = calculate_trading_metrics(trades)
            print(f"Agent {agent_id}:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        else:
            print(f"Agent {agent_id}: No trades executed.")


if __name__ == "__main__":
    # ---- 1. Training ----
    train_base = TradingEnvironment(start_date="2023-01-01", end_date="2026-06-30")
    train_env = CompetitiveTradingEnv(train_base,
                                      reward_mode='independent',   # <-- independent rewards
                                      include_opponent_position=True,
                                      transaction_cost=0.0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = CompetitivePPOTrainer(train_env, lr=3e-4, device=device,
                                    beta=0.1)   # <-- competition coefficient

    print("Starting training with mixed reward (beta=0.1)...")
    pnl_log = trainer.train(total_steps=50000, rollout_steps=2048,
                            save_path='models/competitive')
    print("Training complete.")

    # ---- 2. Evaluation ----
    test_base = TradingEnvironment(start_date="2024-07-01", end_date="2026-12-31")
    test_env = CompetitiveTradingEnv(test_base,
                                     reward_mode='independent',   # evaluate own profit
                                     include_opponent_position=True,
                                     transaction_cost=0.0)

    print("\nEvaluating on test data...")
    evaluate_and_plot(test_env, trainer, max_steps=1000)