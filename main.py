import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict

from training.rl_trainer import RLTrainer
from training.imitation_learning import ImitationLearner
from deployment.live_trader import LiveTrader
from utils.logger import setup_logger
from data.data_loader import DataLoader

def load_config(config_path: str = "config/config.yaml") -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="Trading RL System")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "deploy", "imitate"],
                       default="train", help="Mode to run")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logger(
        level=config['logging']['level'],
        log_file="logs/trading_system.log"
    )
    
    logger.info(f"Starting Trading RL System in {args.mode} mode")
    
    if args.mode == "train":
        # Train RL agent
        trainer = RLTrainer(config['rl'])
        algo, checkpoint = trainer.train(
            stop_iters=config['rl']['training'].get('num_iterations', 100),
            stop_timesteps=config['rl']['training'].get('total_timesteps', 1000000)
        )
        logger.info(f"Training complete. Checkpoint saved to {checkpoint}")
        
    elif args.mode == "imitate":
        # Imitation learning from expert
        learner = ImitationLearner(config['imitation'])
        model = learner.train()
        learner.save_model("models/imitation_policy.pth")
        logger.info("Imitation learning complete")
        
    elif args.mode == "evaluate":
        # Evaluate trained policy
        if not args.checkpoint:
            raise ValueError("Checkpoint path required for evaluation")
        trainer = RLTrainer(config['rl'])
        mean_reward = trainer.evaluate(args.checkpoint)
        logger.info(f"Evaluation complete. Mean reward: {mean_reward:.2f}")
        
    elif args.mode == "deploy":
        # Deploy for live trading
        if not args.checkpoint:
            raise ValueError("Checkpoint path required for deployment")
        trader = LiveTrader(config, args.checkpoint)
        trader.run()
    elif args.mode == "imitate":
        # Imitation learning from expert
        from training.imitation_learning import ImitationLearner
        
        # Load configuration
        imitation_config = config['imitation']
        
        # Set input dimension based on feature engineering
        feature_config = config['data']['feature_columns']
        lookback = config['data']['lookback_window']
        imitation_config['input_dim'] = len(feature_config) * (lookback + 1)
        
        # Create imitation learner
        learner = ImitationLearner(imitation_config)
        
        # Load data
        from data.data_loader import DataLoader
        data_loader = DataLoader(config['data'])
        df = data_loader.load_historical_data()
        
        # Train model
        model = learner.train(df=df)
        
        # Evaluate
        test_df = data_loader.load_test_data()
        metrics = learner.evaluate(test_df)
        
        # Save model
        learner.save_model("models/imitation_policy.pth")
        
        logger.info(f"Imitation learning complete. Validation accuracy: {metrics.get('accuracy', 0):.2f}%")


        
if __name__ == "__main__":
    main()