import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class PositionAwareNetwork(pl.LightningModule):
    """
    Problem with our previous approach:
    When we flatten [temp1, temp2, temp3, temp4, temp5] into one vector,
    the network doesn't know which temperature came WHEN.

    temp1 could be yesterday, temp5 could be 5 days ago - but the network
    treats them all the same!

    Solution: Tell the network WHERE each number came from.
    """

    def __init__(self, input_size=1, hidden_size=32, sequence_length=5, output_size=1, learning_rate=0.01):
        super().__init__()

        # Store these for later use in forward()
        self.sequence_length = sequence_length  # How many numbers in each sequence (5 temperatures)
        self.input_size = input_size  # How many features each number has (1 = just temperature)
        self.hidden_size = hidden_size  # How many neurons in hidden layers (32)

        # Before: each input was just [temperature] - size 1
        # Now: each input will be [temperature, position] - size 2
        # So input_size + 1 = 1 + 1 = 2

        # This layer takes [temperature, position] and converts it to hidden_size features
        # It's like saying: "Given a temperature and when it occurred,
        # create 32 different features that represent this information"
        self.input_processor = nn.Linear(input_size + 1, hidden_size)

        # Why input_size + 1?
        # input_size = 1 (just temperature)
        # +1 = position information (0, 1, 2, 3, 4)
        # So total input to this layer = 2 numbers

        # After processing each position separately, we'll have:
        # Position 0: 32 features
        # Position 1: 32 features
        # Position 2: 32 features
        # Position 3: 32 features
        # Position 4: 32 features
        # Total: 5 × 32 = 160 features

        # This network takes all 160 features and processes them to make final prediction
        self.sequence_processor = nn.Sequential(
            # First layer: 160 features → 32 features
            nn.Linear(hidden_size * sequence_length, hidden_size),  # 32 * 5 = 160 → 32
            nn.ReLU(),  # Activation function (sets negative values to 0)
            # Second layer: 32 features → 32 features
            nn.Linear(hidden_size, hidden_size),  # 32 → 32
            nn.ReLU(),  # Another activation
            # Final layer: 32 features → 1 output (our prediction)
            nn.Linear(hidden_size, output_size),  # 32 → 1
        )

        self.learning_rate = learning_rate

        print(f"Created position-aware network:")
        print(f"Each input: {input_size} number(s) + 1 position → {hidden_size} features")
        print(f"Sequence: {sequence_length} positions × {hidden_size} features = {sequence_length * hidden_size}")

    def forward(self, x):
        """
        This is where the magic happens! Let's trace through exactly what happens
        to our data as it flows through the network.

        INPUT: x has shape [batch_size, sequence_length, input_size]

        Let's say:
        - batch_size = 1 (processing 1 example)
        - sequence_length = 5 (sequence of 5 temperatures)
        - input_size = 1 (each temperature is 1 number)

        So x looks like: [[[20.1], [21.5], [22.3], [21.8], [20.9]]]
        Shape: [1, 5, 1]
        """

        # Extract the dimensions from our input tensor
        batch_size, seq_len, input_size = x.shape
        print(f"\nProcessing input with shape: batch_size={batch_size}, seq_len={seq_len}, input_size={input_size}")

        # STEP 1: Create position encodings
        # We want to add position information to each temperature
        # Position 0 gets number 0.0, position 1 gets 1.0, etc.

        # torch.arange(seq_len) creates: [0, 1, 2, 3, 4]
        # dtype=torch.float32 makes them decimal numbers: [0.0, 1.0, 2.0, 3.0, 4.0]
        # device=x.device ensures they're on same device (CPU/GPU) as our input
        positions = torch.arange(seq_len, dtype=torch.float32, device=x.device)
        print(f"Created position numbers: {positions}")

        # Current shape of positions: [5]
        # We need to reshape it to match our data structure

        # .unsqueeze(0) adds a dimension at position 0: [5] → [1, 5]
        # .unsqueeze(-1) adds a dimension at the end: [1, 5] → [1, 5, 1]
        positions = positions.unsqueeze(0).unsqueeze(-1)
        print(f"Reshaped positions to: {positions.shape}")

        # Now we need to expand this to match our batch size
        # If we're processing multiple examples at once, each needs the same position info
        # .expand(batch_size, -1, -1) copies the position info for each example in the batch
        # -1 means "keep the existing size for this dimension"
        positions = positions.expand(batch_size, -1, -1)
        print(f"Expanded positions for batch: {positions.shape}")

        # STEP 2: Combine temperature data with position data
        # Before: x = [[[20.1], [21.5], [22.3], [21.8], [20.9]]]
        # positions = [[[0.0], [1.0], [2.0], [3.0], [4.0]]]
        # After concatenation: [[[20.1, 0.0], [21.5, 1.0], [22.3, 2.0], [21.8, 3.0], [20.9, 4.0]]]

        # torch.cat() concatenates tensors along a specified dimension
        # dim=-1 means concatenate along the last dimension (the feature dimension)
        x_with_positions = torch.cat([x, positions], dim=-1)
        print(f"Combined data shape: {x_with_positions.shape}")
        print(f"First few values: {x_with_positions[0, :3, :]}")  # Show first 3 positions

        # STEP 3: Process each position separately
        # We're going to take each [temperature, position] pair and convert it to features

        position_features = []  # List to store the features from each position

        for pos in range(seq_len):  # pos will be 0, 1, 2, 3, 4
            print(f"\nProcessing position {pos}:")

            # Extract the data for this specific position across all examples in the batch
            # x_with_positions[:, pos, :] means:
            # : = take all examples in the batch
            # pos = take this specific position (0, 1, 2, 3, or 4)
            # : = take all features (temperature + position)
            pos_input = x_with_positions[:, pos, :]
            print(f"  Input at position {pos}: {pos_input}")
            print(f"  Shape: {pos_input.shape}")  # Should be [batch_size, 2]

            # Pass this through our input processor
            # This linear layer takes [temperature, position] and creates hidden_size features
            # So [20.1, 0.0] becomes [f1, f2, f3, ..., f32] where each f is a feature
            processed = self.input_processor(pos_input)  # Shape: [batch_size, hidden_size]
            print(f"  After linear layer: shape {processed.shape}")

            # Apply ReLU activation (sets negative values to 0)
            pos_features = F.relu(processed)
            print(f"  After ReLU: shape {pos_features.shape}")
            print(f"  First few features: {pos_features[0, :5]}")  # Show first 5 features

            # Add these features to our list
            position_features.append(pos_features)

        # STEP 4: Combine all position features
        # Now we have:
        # position_features[0] = features from position 0 (shape: [batch_size, 32])
        # position_features[1] = features from position 1 (shape: [batch_size, 32])
        # position_features[2] = features from position 2 (shape: [batch_size, 32])
        # position_features[3] = features from position 3 (shape: [batch_size, 32])
        # position_features[4] = features from position 4 (shape: [batch_size, 32])

        # We want to concatenate them all together into one big feature vector
        # torch.cat(list, dim=1) concatenates along dimension 1 (the feature dimension)
        all_features = torch.cat(position_features, dim=1)
        print(f"\nCombined all features: shape {all_features.shape}")  # Should be [batch_size, 160]

        # STEP 5: Final processing
        # Pass the combined features through our sequence processor network
        # This takes all 160 features and processes them to make the final prediction
        final_output = self.sequence_processor(all_features)
        print(f"Final output: shape {final_output.shape}, value {final_output}")

        return final_output

    def training_step(self, batch, batch_idx):
        """
        This is called during training for each batch of data.

        batch contains:
        - x: input sequences [batch_size, sequence_length, input_size]
        - y: target outputs [batch_size, output_size]
        """
        x, y = batch

        # Get prediction from our network
        prediction = self(x)  # This calls forward() method above

        # Calculate how wrong we were (mean squared error)
        loss = F.mse_loss(prediction, y)

        # Log the loss so we can track training progress
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        """
        This tells PyTorch Lightning what optimizer to use for training.
        Adam is a popular optimizer that adjusts the network weights to minimize loss.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def create_position_sensitive_data():
    """
    Let's create data where POSITION really matters.

    Task: Predict if the temperature trend is going UP or DOWN
    - If recent temperatures are higher than older ones → 1 (UP)
    - If recent temperatures are lower than older ones → 0 (DOWN)

    This task REQUIRES knowing which temperature came when!
    """
    print("Creating position-sensitive data...")
    print("Task: Determine if temperature trend is UP (1) or DOWN (0)")

    sequences = []  # Will store our input sequences
    labels = []  # Will store our target labels (0 or 1)

    # Create 1000 examples
    for example_num in range(1000):
        # Start with a random base temperature between 10-30°C
        base_temp = torch.rand(1) * 20 + 10  # torch.rand(1) gives random number 0-1

        # Randomly decide if this example should be UP trend or DOWN trend
        if torch.rand(1) > 0.5:  # 50% chance
            # CREATE AN UP TREND
            trend_direction = 1  # Label = 1 means UP

            # Add some random noise to make it realistic
            noise = torch.randn(5) * 1.0  # 5 random numbers from normal distribution

            # Create an upward trend: temperatures gradually increase
            # torch.linspace(0, 3, 5) creates [0.0, 0.75, 1.5, 2.25, 3.0]
            trend = torch.linspace(0, 3, 5)

            # Combine base temperature + upward trend + noise
            temps = base_temp + trend + noise

        else:
            # CREATE A DOWN TREND
            trend_direction = 0  # Label = 0 means DOWN

            noise = torch.randn(5) * 1.0

            # Create a downward trend: temperatures gradually decrease
            # torch.linspace(0, -3, 5) creates [0.0, -0.75, -1.5, -2.25, -3.0]
            trend = torch.linspace(0, -3, 5)

            # Combine base temperature + downward trend + noise
            temps = base_temp + trend + noise

        # Add feature dimension: [5] → [5, 1]
        # This makes each temperature a "feature vector" of size 1
        sequences.append(temps.unsqueeze(-1))

        # Store the label (0 for DOWN, 1 for UP)
        labels.append(torch.tensor([trend_direction], dtype=torch.float32))

    # Convert lists to tensors
    sequences = torch.stack(sequences)  # Shape: [1000, 5, 1]
    labels = torch.stack(labels)  # Shape: [1000, 1]

    print(f"Created {len(sequences)} sequences")

    # Show examples
    up_examples = sequences[labels.squeeze() == 1]  # Get UP trend examples
    down_examples = sequences[labels.squeeze() == 0]  # Get DOWN trend examples

    print("Example UP trend:", [f"{x:.1f}" for x in up_examples[0].squeeze().tolist()])
    print("Example DOWN trend:", [f"{x:.1f}" for x in down_examples[0].squeeze().tolist()])

    return sequences, labels


if __name__ == "__main__":
    # Let's test our position-aware network with a simple example
    print("=" * 70)
    print("TESTING POSITION-AWARE NETWORK")
    print("=" * 70)

    # Create a simple example
    # One sequence: temperatures [20.0, 21.0, 22.0, 23.0, 24.0] (clear UP trend)
    example_sequence = torch.tensor([[[20.0], [21.0], [22.0], [23.0], [24.0]]])
    print(f"Example input sequence: {example_sequence.squeeze().tolist()}")

    # Create the network
    model = PositionAwareNetwork(
        input_size=1,  # Each temperature is 1 number
        sequence_length=5,  # 5 temperatures in sequence
        hidden_size=8,  # Use smaller hidden size for clearer debugging
        output_size=1,  # Predict 1 value
    )

    # Process the example (this will show all the detailed steps)
    print("\n" + "=" * 50)
    print("PROCESSING EXAMPLE THROUGH NETWORK:")
    print("=" * 50)

    with torch.no_grad():  # Don't compute gradients (we're just testing)
        result = model(example_sequence)

    print(f"\nFinal result: {result.item():.4f}")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print("1. Each temperature gets combined with its position number")
    print("2. [temp, position] pairs are processed separately for each position")
    print("3. All position features are combined into one big vector")
    print("4. The final network processes all features to make a prediction")
    print("5. The network now KNOWS which temperature came when!")
