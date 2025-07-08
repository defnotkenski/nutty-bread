import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class SequenceNetwork(pl.LightningModule):
    """
    Up until now, we've been working with just ONE number at a time.
    But what if we want to work with MULTIPLE numbers together?
    
    For example:
    - A sentence has multiple words
    - A song has multiple notes  
    - A day has multiple temperature readings
    
    This is called working with "sequences" - lists of related numbers.
    """
    
    def __init__(self, input_size=1, hidden_size=32, sequence_length=5, output_size=1, learning_rate=0.01):
        super().__init__()
        
        self.sequence_length = sequence_length
        
        # Now our input is sequence_length * input_size numbers!
        # If sequence_length=5 and input_size=1, we expect 5 numbers as input
        flattened_input_size = sequence_length * input_size
        
        self.network = nn.Sequential(
            nn.Linear(flattened_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        self.learning_rate = learning_rate
        
        print(f"Created sequence network:")
        print(f"Input: {sequence_length} numbers → Flattened to {flattened_input_size} → Hidden({hidden_size}) → Output({output_size})")
        
    def forward(self, x):
        """
        x comes in as shape [batch_size, sequence_length, input_size]
        We need to flatten it to [batch_size, sequence_length * input_size]
        
        Think of it like taking [1.2, 3.4, 5.6, 7.8, 9.0] 
        and treating it as one long input vector.
        """
        batch_size = x.size(0)
        # Flatten the sequence: [batch, seq_len, input_size] → [batch, seq_len * input_size]
        x_flattened = x.view(batch_size, -1)
        return self.network(x_flattened)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        loss = F.mse_loss(prediction, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

def create_sequence_data():
    """
    Let's create data where we have SEQUENCES of numbers.
    
    Our task: Given the last 5 temperature readings, predict tomorrow's temperature.
    """
    print("Creating sequence data...")
    print("Task: Given 5 consecutive temperature readings, predict the next one")
    
    # Create a long temperature series (like daily temperatures over time)
    time_steps = torch.linspace(0, 50, 1000)  # 1000 days
    
    # Temperature pattern: seasonal cycle + some randomness
    temperatures = 20 + 10 * torch.sin(time_steps * 0.1) + torch.randn(1000) * 2
    
    sequences = []
    targets = []
    
    # Create sequences: [day1, day2, day3, day4, day5] → day6
    sequence_length = 5
    
    for i in range(len(temperatures) - sequence_length):
        # Take 5 consecutive temperatures
        seq = temperatures[i:i+sequence_length]
        # Target is the next temperature
        target = temperatures[i+sequence_length]
        
        sequences.append(seq)
        targets.append(target)
    
    # Convert to tensors and add the right dimensions
    sequences = torch.stack(sequences).unsqueeze(-1)  # [num_sequences, 5, 1]
    targets = torch.stack(targets).unsqueeze(-1)      # [num_sequences, 1]
    
    print(f"Created {len(sequences)} sequences")
    print(f"Each sequence has {sequences.shape[1]} numbers")
    print(f"Example sequence: {sequences[0].squeeze().tolist()}")
    print(f"Example target: {targets[0].item():.1f}")
    
    return sequences, targets

def demonstrate_sequence_network():
    """
    Let's see how a network can work with sequences of numbers.
    """
    print("=" * 60)
    print("WORKING WITH SEQUENCES OF NUMBERS")
    print("=" * 60)
    
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create sequence data
    sequences, targets = create_sequence_data()
    
    # Split data
    train_size = int(0.8 * len(sequences))
    train_sequences = sequences[:train_size]
    train_targets = targets[:train_size]
    test_sequences = sequences[train_size:]
    test_targets = targets[train_size:]
    
    train_dataset = TensorDataset(train_sequences, train_targets)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Create sequence network
    model = SequenceNetwork(
        input_size=1,        # Each temperature is 1 number
        sequence_length=5,   # We look at 5 temperatures at once
        hidden_size=64,
        output_size=1        # Predict 1 temperature
    )
    
    print(f"\n🔧 Training network to predict temperatures...")
    trainer = pl.Trainer(max_epochs=50, enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model, train_dataloader)
    
    # Test the model
    print(f"\n📊 TESTING PREDICTIONS:")
    print("Input: [temp1, temp2, temp3, temp4, temp5] → Predicted → Actual")
    print("-" * 65)
    
    model.eval()
    with torch.no_grad():
        for i in range(0, min(10, len(test_sequences)), 2):  # Show every 2nd example
            input_seq = test_sequences[i]
            actual_target = test_targets[i]
            predicted = model(input_seq.unsqueeze(0))  # Add batch dimension
            
            seq_values = [f"{x:.1f}" for x in input_seq.squeeze().tolist()]
            print(f"[{', '.join(seq_values)}] → {predicted.item():.1f}°C → {actual_target.item():.1f}°C")
    
    # Calculate error
    test_predictions = model(test_sequences)
    test_error = F.mse_loss(test_predictions, test_targets)
    print(f"\nTest Error: {test_error.item():.3f}")
    
    print(f"\n💡 Notice: The network learned to use ALL 5 temperatures together!")
    print(f"   But it treats them as one big flattened input: [t1, t2, t3, t4, t5]")
    print(f"   It doesn't know that t1 came before t2, or that t5 is most recent.")

if __name__ == "__main__":
    demonstrate_sequence_network()
    
    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("="*60)
    print("1. We can work with SEQUENCES of numbers, not just single numbers")
    print("2. We flatten the sequence into one long input vector")
    print("3. The network learns patterns across the whole sequence")
    print("4. But we lose information about the ORDER of the numbers")
    print("\n🤔 PROBLEM: What if the ORDER matters?")
    print("   What if we want the network to know which number came first?")
    print("   That's our next challenge...")
