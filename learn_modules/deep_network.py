import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class DeepNetwork(pl.LightningModule):
    """
    Now we have MANY layers stacked together.
    This is called a "deep" neural network (hence "deep learning").
    
    Each layer learns to detect different patterns:
    - Layer 1 might learn simple features
    - Layer 2 might combine those into more complex features  
    - Layer 3 might combine those into even more complex patterns
    - And so on...
    """
    
    def __init__(self, input_size=1, hidden_size=32, num_layers=4, output_size=1, learning_rate=0.01):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create a list to hold all our layers
        layers = []
        
        # First layer: input_size → hidden_size
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        # Middle/hidden layers: hidden_size → hidden_size
        for i in range(num_layers - 2):  # -2 because we already have first and will add last
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        # Final layer: hidden_size → output_size
        layers.append(nn.Linear(hidden_size, output_size))
        
        # Combine all layers into one sequential model
        self.network = nn.Sequential(*layers)
        
        self.learning_rate = learning_rate
        
        print(f"Created deep network with {num_layers} layers:")
        print(f"Input({input_size}) → Hidden({hidden_size}) → ... → Hidden({hidden_size}) → Output({output_size})")
        
    def forward(self, x):
        """
        Now the data flows through many layers:
        Input → Layer1 → ReLU → Layer2 → ReLU → ... → FinalLayer → Output
        
        Each layer transforms the data a little bit more.
        """
        return self.network(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        loss = F.mse_loss(prediction, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

def create_complex_data():
    """
    Let's create a really complex pattern that would be hard for 
    a shallow network to learn.
    
    This combines multiple sine waves - a very wiggly pattern!
    """
    x = torch.linspace(-3, 3, 400).unsqueeze(1)  # 400 points from -3 to 3
    
    # Complex pattern: combination of sine waves
    y = torch.sin(x) + 0.5 * torch.sin(3 * x) + 0.3 * torch.sin(5 * x)
    
    # Add a tiny bit of noise
    y = y + torch.randn_like(y) * 0.05
    
    return x, y

def compare_networks():
    """
    Let's compare shallow vs deep networks on the same complex data.
    """
    print("=" * 50)
    print("COMPARING SHALLOW VS DEEP NETWORKS")
    print("=" * 50)
    
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create complex data
    x_data, y_data = create_complex_data()
    dataset = TensorDataset(x_data, y_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"Data has {len(x_data)} points with a very wiggly pattern")
    
    # Create shallow network (2 layers, wide)
    shallow_model = DeepNetwork(
        input_size=1, 
        hidden_size=64,  # Wider to compensate for being shallow
        num_layers=2, 
        output_size=1
    )
    
    # Create deep network (6 layers, narrower)  
    deep_model = DeepNetwork(
        input_size=1,
        hidden_size=32,  # Narrower but deeper
        num_layers=6,
        output_size=1
    )
    
    # Train shallow network
    print("\n🔍 Training SHALLOW network (2 layers, 64 neurons each)...")
    shallow_trainer = pl.Trainer(max_epochs=100, enable_progress_bar=False, enable_model_summary=False)
    shallow_trainer.fit(shallow_model, dataloader)
    
    # Train deep network
    print("\n🧠 Training DEEP network (6 layers, 32 neurons each)...")
    deep_trainer = pl.Trainer(max_epochs=100, enable_progress_bar=False, enable_model_summary=False)
    deep_trainer.fit(deep_model, dataloader)
    
    # Test both networks
    test_x = torch.linspace(-3, 3, 20).unsqueeze(1)
    
    print("\n📊 COMPARISON RESULTS:")
    print("Input → Shallow Prediction vs Deep Prediction")
    print("-" * 45)
    
    for i in range(0, len(test_x), 3):  # Test every 3rd point
        x_val = test_x[i]
        true_y = torch.sin(x_val) + 0.5 * torch.sin(3 * x_val) + 0.3 * torch.sin(5 * x_val)
        
        shallow_pred = shallow_model(x_val)
        deep_pred = deep_model(x_val)
        
        print(f"x={x_val.item():+.1f} → Shallow: {shallow_pred.item():+.2f} | Deep: {deep_pred.item():+.2f} | True: {true_y.item():+.2f}")
    
    # Calculate overall error
    with torch.no_grad():
        shallow_error = F.mse_loss(shallow_model(test_x), 
                                 torch.sin(test_x) + 0.5 * torch.sin(3 * test_x) + 0.3 * torch.sin(5 * test_x))
        deep_error = F.mse_loss(deep_model(test_x),
                               torch.sin(test_x) + 0.5 * torch.sin(3 * test_x) + 0.3 * torch.sin(5 * test_x))
    
    print(f"\n📈 FINAL SCORES:")
    print(f"Shallow network error: {shallow_error.item():.4f}")
    print(f"Deep network error: {deep_error.item():.4f}")
    
    if deep_error < shallow_error:
        print("🏆 Deep network wins! It learned the complex pattern better.")
    else:
        print("🏆 Shallow network wins! Sometimes simple is better.")

if __name__ == "__main__":
    compare_networks()
    
    print("\n" + "="*50)
    print("KEY INSIGHTS ABOUT DEEP NETWORKS:")
    print("="*50)
    print("1. More layers = can learn more complex patterns")
    print("2. Each layer builds on what previous layers learned")
    print("3. Early layers learn simple features, later layers learn complex combinations")
    print("4. This is why it's called 'deep learning'")
    print("5. But deeper isn't always better - depends on your data!")
