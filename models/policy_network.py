import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

class TradingPolicyNetwork(nn.Module):
    """Advanced policy network with attention mechanism (LayerNorm version)"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        output_dim: int = 4,
        use_attention: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        self.use_attention = use_attention
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build network layers using LayerNorm instead of BatchNorm
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),      # ← works with batch size 1
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
            
        self.feature_layers = nn.Sequential(*layers)
        
        # Attention mechanism on the final hidden dimension
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dims[-1],     # ← use last dimension, e.g., 64
                num_heads=4,
                dropout=dropout_rate,
                batch_first=True
            )
            self.attn_layer_norm = nn.LayerNorm(hidden_dims[-1])
        
        # Output heads
        self.logits_head = nn.Linear(prev_dim, output_dim)
        self.value_head = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(
        self, 
        obs: torch.Tensor, 
        state: Optional[Dict] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Forward pass"""
        batch_size = obs.shape[0]
        
        # Extract features
        features = self.feature_layers(obs)
        
        # Apply attention if enabled and batch size > 1
        if self.use_attention and batch_size > 1:
            # Reshape for attention (batch, seq_len=1, features)
            features = features.unsqueeze(1)
            attn_out, _ = self.attention(features, features, features)
            features = self.attn_layer_norm(features + attn_out)
            features = features.squeeze(1)
        
        # Compute logits and value
        logits = self.logits_head(features)
        value = self.value_head(features)
        
        # Apply action mask if provided
        if mask is not None:
            logits = logits + (1 - mask) * -1e9
            
        # Get action distribution
        probs = F.softmax(logits, dim=-1)
        
        return logits, value, {'probs': probs, 'features': features}
    
    def get_action(
        self, 
        obs: np.ndarray, 
        deterministic: bool = False,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[int, Dict]:
        """Get action from a single observation"""
        self.eval()  # ensure no training-specific behaviour (though LayerNorm is fine)
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            mask_tensor = torch.FloatTensor(mask).unsqueeze(0) if mask is not None else None
            
            logits, _, info = self.forward(obs_tensor, mask=mask_tensor)
            probs = info['probs'].squeeze(0).cpu().numpy()
            
            if deterministic:
                action = np.argmax(probs)
            else:
                action = np.random.choice(len(probs), p=probs)
                
        self.train()  # return to training mode if needed
        return action, {'probs': probs, 'action_value': action}
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING POLICY NETWORK")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Using device: {device}")
    
    # Test 1: Create network
    print("\n🏗️ Test 1: Creating policy network...")
    input_dim = 100
    hidden_dims = [256, 128, 64]
    output_dim = 4
    
    model = TradingPolicyNetwork(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        use_attention=True
    ).to(device)
    
    print(f"✓ Network created")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden layers: {hidden_dims}")
    print(f"  Output dimension: {output_dim}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test 2: Forward pass (batch)
    print("\n🧠 Test 2: Testing forward pass...")
    batch_size = 32
    x = torch.randn(batch_size, input_dim).to(device)
    
    logits, value, info = model(x)
    print(f"✓ Forward pass successful")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Value shape: {value.shape}")
    print(f"  Probs shape: {info['probs'].shape}")
    
    # Test 3: Action selection (single observation)
    print("\n🎮 Test 3: Testing action selection...")
    obs = np.random.randn(input_dim).astype(np.float32)
    
    # Deterministic action
    action, info_det = model.get_action(obs, deterministic=True)
    print(f"  Deterministic action: {action}")
    print(f"  Action probabilities: {info_det['probs']}")
    
    # Stochastic action (multiple samples)
    actions = []
    for _ in range(100):
        action, _ = model.get_action(obs, deterministic=False)
        actions.append(action)
    
    action_counts = np.bincount(actions, minlength=output_dim)
    action_probs = action_counts / 100
    print(f"  Stochastic action distribution: {action_probs}")
    
    # Test 4: Test with action mask
    print("\n🎭 Test 4: Testing action masking...")
    mask = np.array([1, 1, 0, 0])  # Only allow BUY and SELL
    
    action, info_masked = model.get_action(obs, deterministic=True, mask=mask)
    print(f"  Mask: {mask}")
    print(f"  Masked action: {action} (should be 0 or 1)")
    print(f"  Masked probabilities: {info_masked['probs']}")
    
    # Test 5: Test batch processing with masks
    print("\n📊 Test 5: Testing batch processing...")
    batch_obs = np.random.randn(16, input_dim).astype(np.float32)
    batch_masks = np.ones((16, output_dim))
    for i in range(16):
        if np.random.random() > 0.5:
            batch_masks[i, 2] = 0  # Mask HOLD for some samples
    
    batch_obs_tensor = torch.FloatTensor(batch_obs).to(device)
    batch_masks_tensor = torch.FloatTensor(batch_masks).to(device)
    
    logits, values, info_batch = model(batch_obs_tensor, mask=batch_masks_tensor)
    print(f"✓ Batch processing successful")
    print(f"  Batch logits shape: {logits.shape}")
    print(f"  Batch values shape: {values.shape}")
    
    # Test 6: Test gradient flow
    print("\n📈 Test 6: Testing gradient flow...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    x_test = torch.randn(8, input_dim).to(device)
    y_test = torch.randint(0, output_dim, (8,)).to(device)
    
    logits, values, _ = model(x_test)
    loss = F.cross_entropy(logits, y_test)
    loss.backward()
    
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    
    print(f"✓ Gradient flow verified")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Mean gradient norm: {np.mean(grad_norms):.6f}")
    
    # Test 7: Test model serialization
    print("\n💾 Test 7: Testing model save/load...")
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=True) as tmpfile:
        # Save model
        torch.save(model.state_dict(), tmpfile.name)
        print(f"✓ Model saved to: {tmpfile.name}")
        
        # Create new model with identical architecture
        new_model = TradingPolicyNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            use_attention=True  # match original model
        ).to(device)
        new_model.load_state_dict(torch.load(tmpfile.name))
        print(f"✓ Model loaded successfully")
        
        # Verify same output (compare only tensors, ignore info dict)
        test_input = torch.randn(1, input_dim).to(device)
        with torch.no_grad():
            logits1, value1, _ = model(test_input)
            logits2, value2, _ = new_model(test_input)
            
        diff_logits = (logits1 - logits2).abs().mean().item()
        diff_value = (value1 - value2).abs().mean().item()
        print(f"  Logits difference after load: {diff_logits:.8f}")
        print(f"  Value difference after load: {diff_value:.8f}")