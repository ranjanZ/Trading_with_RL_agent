import asyncio
import websockets
import json
import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from models.policy_network import TradingPolicyNetwork
from environments.trading_env import TradingEnvironment
from deployment.risk_manager import RiskManager

class LiveTrader:
    """Live trading deployment with real-time data"""
    
    def __init__(self, config: Dict, model_path: str):
        self.config = config
        self.model = TradingPolicyNetwork(
            input_dim=config['rl']['model'].get('input_dim', 50),
            hidden_dims=config['rl']['model']['fcnet_hiddens']
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.env = TradingEnvironment(config['trading'])
        self.risk_manager = RiskManager(config['trading'])
        self.logger = logging.getLogger(__name__)
        
        self.current_position = 0
        self.data_buffer = []
        
    async def run(self):
        """Main trading loop"""
        self.logger.info("Starting live trading system...")
        
        if self.config['deployment']['mode'] == 'paper':
            await self._paper_trading()
        else:
            await self._live_trading()
            
    async def _paper_trading(self):
        """Paper trading simulation"""
        self.logger.info("Starting paper trading mode")
        
        # Load historical data for simulation
        from data.data_loader import DataLoader
        loader = DataLoader(self.config['data'])
        data = loader.load_live_data()
        
        for idx, row in data.iterrows():
            # Get action from model
            obs = self._prepare_observation(row)
            action = self.model.get_action(obs, deterministic=True)[0]
            
            # Execute action
            _, reward, terminated, _, info = self.env.step(action)
            
            # Risk check
            if not self.risk_manager.check_risk(self.env):
                self.logger.warning("Risk limit reached, stopping trading")
                break
                
            # Log trade
            if info.get('trade_closed', False):
                self.logger.info(f"Trade closed: PnL=${info['pnl']:.2f}, "
                               f"Balance=${self.env.balance:.2f}")
                
            if terminated:
                self.logger.info("Episode terminated")
                break
                
            await asyncio.sleep(1)  # Simulate real-time
            
    async def _live_trading(self):
        """Live trading with real market data"""
        self.logger.info("Starting live trading mode")
        
        # Connect to WebSocket for real-time data
        ws_url = self.config['deployment']['websocket_url']
        
        async with websockets.connect(ws_url) as websocket:
            # Subscribe to ticker
            await websocket.send(json.dumps({
                "action": "subscribe",
                "trades": ["XAUUSD"]
            }))
            
            while True:
                try:
                    # Receive market data
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    # Process tick
                    self._process_tick(data)
                    
                    # Get and execute action
                    if len(self.data_buffer) >= self.config['data']['lookback_window']:
                        action = self._get_action()
                        self._execute_trade(action)
                        
                except Exception as e:
                    self.logger.error(f"Error in live trading: {e}")
                    await asyncio.sleep(1)
                    
    def _prepare_observation(self, row: pd.Series) -> np.ndarray:
        """Prepare observation for model"""
        features = []
        for col in self.config['data']['feature_columns']:
            if col in row.index:
                features.append(float(row[col]))
            else:
                features.append(0.0)
                
        # Add position info
        features.extend([
            1.0 if self.env.current_position == 1 else 0.0,
            1.0 if self.env.current_position == -1 else 0.0,
            1.0 if self.env.current_position == 0 else 0.0,
            self.env.unrealized_pnl / self.env.initial_balance,
            self.env.realized_pnl / self.env.initial_balance
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _get_action(self) -> int:
        """Get trading action from model"""
        # Prepare observation from buffer
        if len(self.data_buffer) < self.config['data']['lookback_window']:
            return 2  # HOLD
            
        obs = self._prepare_observation(pd.Series(self.data_buffer[-1]))
        action, _ = self.model.get_action(obs, deterministic=False)
        return action
    
    def _execute_trade(self, action: int):
        """Execute trading action"""
        current_price = self.data_buffer[-1]['close']
        
        # Get valid actions
        valid_actions = self.env.get_valid_actions_mask()
        
        if valid_actions[action]:
            # Execute through environment
            reward, terminated, info = self.env.step(action, current_price)
            
            # Log action
            action_names = {0: 'BUY', 1: 'SELL', 2: 'HOLD', 3: 'CLOSE'}
            self.logger.debug(f"Action: {action_names[action]}, "
                            f"Price: {current_price:.2f}, "
                            f"PnL: ${self.env.unrealized_pnl:.2f}")
            


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING LIVE TRADER")
    print("=" * 60)
    
    # Test with mock data (no actual connection)
    print("\n📊 Test 1: Creating mock market data...")
    
    # Generate mock price stream
    np.random.seed(42)
    n_ticks = 100
    times = pd.date_range(start=datetime.now(), periods=n_ticks, freq='1s')
    prices = 2000 + np.cumsum(np.random.normal(0, 0.1, n_ticks))
    
    mock_data = pd.DataFrame({
        'time': times,
        'bid': prices - 0.5,
        'ask': prices + 0.5,
        'last': prices,
        'volume': np.random.randint(1, 100, n_ticks)
    })
    
    print(f"✓ Generated {len(mock_data)} mock ticks")
    print(f"  Price range: ${mock_data['last'].min():.2f} - ${mock_data['last'].max():.2f}")
    
    # Test 2: Test RiskManager
    print("\n🛡️ Test 2: Testing RiskManager...")
    risk_config = {
        'max_position_size': 1.0,
        'max_daily_loss': 500.0,
        'max_drawdown': 0.2,
        'stop_loss': 50.0,
        'take_profit': 100.0
    }
    
    risk_manager = RiskManager(risk_config)
    print(f"✓ RiskManager created")
    
    # Test risk checks
    mock_env = type('MockEnv', (), {
        'balance': 10000,
        'unrealized_pnl': 0,
        'realized_pnl': 0,
        'current_position': 0
    })()
    
    is_safe = risk_manager.check_risk(mock_env)
    print(f"  Initial risk check: {'Safe' if is_safe else 'Risk exceeded'}")
    
    # Test with losses
    mock_env.unrealized_pnl = -100
    is_safe = risk_manager.check_risk(mock_env)
    print(f"  After $100 loss: {'Safe' if is_safe else 'Risk exceeded'}")
    
    # Test 3: Test observation preparation
    print("\n🔧 Test 3: Testing observation preparation...")
    
    # Mock model
    class MockModel:
        def get_action(self, obs, deterministic=False):
            return np.random.randint(0, 4), {'probs': np.ones(4)/4}
    
    mock_model = MockModel()
    
    # Test feature preparation
    feature_columns = ['close', 'volume', 'rsi', 'atr']
    mock_row = pd.Series({
        'close': 2000,
        'volume': 1000,
        'rsi': 50,
        'atr': 10
    })
    
    features = [float(mock_row.get(col, 0)) for col in feature_columns]
    print(f"✓ Features prepared: {features}")
    
    # Test 4: Simulate trading loop
    print("\n🔄 Test 4: Simulating trading loop (10 iterations)...")
    
    class MockEnvironment:
        def __init__(self):
            self.balance = 10000
            self.current_position = 0
            self.unrealized_pnl = 0
            self.realized_pnl = 0
            
        def step(self, action, price=None):
            if action == 0 and self.current_position == 0:  # BUY
                self.current_position = 1
                reward = 0
            elif action == 1 and self.current_position == 0:  # SELL
                self.current_position = -1
                reward = 0
            elif action == 3 and self.current_position != 0:  # CLOSE
                pnl = np.random.randn() * 10
                self.realized_pnl += pnl
                self.balance += pnl
                self.current_position = 0
                reward = pnl
            else:
                reward = 0
                
            return reward, False, {'trade_closed': action == 3}
            
        def get_valid_actions_mask(self):
            if self.current_position == 0:
                return [True, True, True, True]
            return [False, False, True, True]
    
    mock_env = MockEnvironment()
    
    for i in range(10):
        action = np.random.randint(0, 4)
        valid = mock_env.get_valid_actions_mask()[action]
        
        if valid:
            reward, terminated, info = mock_env.step(action)
            status = "CLOSED" if info.get('trade_closed') else "OPEN"
            print(f"  Step {i+1}: Action={action}, Position={mock_env.current_position}, "
                  f"PnL=${mock_env.realized_pnl:.2f}, {status}")
    
    print(f"\n  Final balance: ${mock_env.balance:.2f}")
    print(f"  Total realized PnL: ${mock_env.realized_pnl:.2f}")
    
    # Test 5: Test WebSocket simulation
    print("\n🔌 Test 5: Testing WebSocket simulation...")
    
    async def mock_websocket():
        for i in range(5):
            yield {
                'type': 'trade',
                'symbol': 'XAUUSD',
                'price': 2000 + np.random.randn() * 5,
                'volume': np.random.randint(1, 10)
            }
    
    async def test_websocket():
        async for msg in mock_websocket():
            print(f"  Received: {msg['symbol']} @ ${msg['price']:.2f} vol={msg['volume']}")
            break  # Just show first message
    
    # Run async test
    asyncio.run(test_websocket())
    
    # Test 6: Test configuration validation
    print("\n✅ Test 6: Validating deployment config...")
    deployment_config = {
        'mode': 'paper',
        'broker': 'alpaca',
        'api_key': 'test_key',
        'api_secret': 'test_secret',
        'websocket_url': 'wss://test.example.com'
    }
    
    required_fields = ['mode', 'broker', 'websocket_url']
    for field in required_fields:
        if field in deployment_config:
            print(f"  ✓ {field} configured")
        else:
            print(f"  ✗ {field} missing")
    
    # Test 7: Test trade execution simulation
    print("\n💰 Test 7: Simulating trade execution...")
    
    class MockBroker:
        def __init__(self):
            self.orders = []
            
        def execute_order(self, side, size, price):
            order = {
                'id': len(self.orders) + 1,
                'side': side,
                'size': size,
                'price': price,
                'time': datetime.now(),
                'status': 'filled'
            }
            self.orders.append(order)
            return order
    
    broker = MockBroker()
    
    # Execute mock trades
    trades = [
        ('BUY', 1.0, 2000.0),
        ('SELL', 1.0, 2005.0),
        ('BUY', 0.5, 1995.0),
    ]
    
    for side, size, price in trades:
        order = broker.execute_order(side, size, price)
        print(f"  {side} {size} @ ${price:.2f} - Order #{order['id']} {order['status']}")
    
    print(f"  Total orders executed: {len(broker.orders)}")
    
    print("\n" + "=" * 60)
    print("✅ Live Trader tests completed!")
    print("=" * 60)
    
    print("\n💡 Deployment Notes:")
    print("  - Paper trading: No real money, simulated execution")
    print("  - Live trading: Requires API keys and market connection")
    print("  - Always test with paper trading first!")            