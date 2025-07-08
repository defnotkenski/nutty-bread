import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class SimpleAttentionNetwork(pl.LightningModule):
    """
    EVOLUTION: From Position-Aware to Attention

    Our previous network had a problem:
    - We processed each position separately: pos0 → features, pos1 → features, etc.
    - Then we just concatenated everything together
    - This is like treating each day's temperature as completely independent!

    But in real life:
    - Yesterday's temperature might be MORE important for today's prediction
    - Or maybe the temperature 3 days ago is key because of a weather pattern
    - We want the network to CHOOSE which temperatures to pay attention to!

    ATTENTION = "Let the network decide what's important"
    """

    def __init__(self, input_size=1, hidden_size=32, sequence_length=5, output_size=1, learning_rate=0.01):
        super().__init__()

        self.sequence_length = sequence_length
        self.input_size = input_size
        self.hidden_size = hidden_size

        # STEP 1: Convert each input to hidden features (same as before)
        # This takes [temperature, position] and makes it into hidden_size features
        self.input_processor = nn.Linear(input_size + 1, hidden_size)

        # STEP 2: THE ATTENTION MECHANISM
        # This is the new magic! The attention mechanism has 3 parts:

        # 2a) QUERY: "What am I looking for?"
        # Think of this as: "What kind of temperature pattern am I trying to find?"
        self.query_layer = nn.Linear(hidden_size, hidden_size)

        # 2b) KEY: "What does each temperature represent?"
        # Think of this as: "How should I describe each temperature's characteristics?"
        self.key_layer = nn.Linear(hidden_size, hidden_size)

        # 2c) VALUE: "What information does each temperature contribute?"
        # Think of this as: "What useful info can I extract from each temperature?"
        self.value_layer = nn.Linear(hidden_size, hidden_size)

        # STEP 3: Final prediction layer
        self.output_layer = nn.Linear(hidden_size, output_size)

        self.learning_rate = learning_rate

    def forward(self, x):
        """
        Let's trace through what happens to our temperature data!

        INPUT: x = [[[20.1], [21.5], [22.3], [21.8], [20.9]]] (5 temperatures)
        Shape: [batch_size, sequence_length, input_size] = [1, 5, 1]
        """

        batch_size, seq_len, input_size = x.shape
        print(f"\n🌡️  Processing {seq_len} temperatures...")

        # STEP 1: Add position information (same as before)
        positions = torch.arange(seq_len, dtype=torch.float32, device=x.device)
        positions = positions.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1)
        x_with_positions = torch.cat([x, positions], dim=-1)
        print(f"Added positions: {x_with_positions.shape}")

        # STEP 2: Convert each temperature to features
        # Instead of processing each position separately, we process ALL at once!

        # WHY WE NEED TO RESHAPE:
        # Our data: [1, 5, 2] = [batch_size, sequence_length, features]
        # But nn.Linear expects: [N, features] where N can be any number
        # So we need to "flatten" the batch and sequence dimensions together

        # .view() method reshapes tensors without changing the data
        # view(-1, input_size + 1) means:
        # -1 = "figure out this dimension automatically" (will be 1*5 = 5)
        # input_size + 1 = keep this dimension as 2
        # Result: [1, 5, 2] → [5, 2]
        x_flat = x_with_positions.view(-1, input_size + 1)  # [5, 2]

        # Pass through linear layer: [5, 2] → [5, hidden_size]
        # self.input_processor is nn.Linear(2, hidden_size)
        # This applies the SAME linear transformation to all 5 temperature+position pairs
        features_flat = self.input_processor(x_flat)  # [5, hidden_size]

        # F.relu() applies ReLU activation function
        # ReLU(x) = max(0, x) - it sets all negative values to 0
        # This adds non-linearity to our network (without it, everything would be just linear math)
        features_flat = F.relu(features_flat)  # [5, hidden_size]

        # Now reshape back to separate batch and sequence dimensions
        # view(batch_size, seq_len, self.hidden_size) = view(1, 5, hidden_size)
        # Result: [5, hidden_size] → [1, 5, hidden_size]
        features = features_flat.view(batch_size, seq_len, self.hidden_size)  # [1, 5, hidden_size]

        # STEP 3: THE ATTENTION MAGIC! 🎭

        # 3a) Create QUERIES: "What am I looking for?"
        # Each temperature asks: "What kind of other temperatures should I pay attention to?"
        queries = self.query_layer(features)  # [1, 5, hidden_size]

        # 3b) Create KEYS: "What do I represent?"
        # Each temperature describes itself: "I am a temperature with these characteristics"
        keys = self.key_layer(features)  # [1, 5, hidden_size]

        # 3c) Create VALUES: "What information can I provide?"
        # Each temperature says: "If you pay attention to me, here's what I'll contribute"
        values = self.value_layer(features)  # [1, 5, hidden_size]

        # STEP 4: CALCULATE ATTENTION SCORES 🎯
        # For each temperature, calculate how much it should pay attention to every other temperature

        # torch.matmul() performs matrix multiplication
        # We multiply queries × keys^T (keys transposed)
        #
        # queries shape: [1, 5, hidden_size]
        # keys shape: [1, 5, hidden_size]
        # keys.transpose(-2, -1) swaps the last two dimensions: [1, 5, hidden_size] → [1, hidden_size, 5]
        #
        # Matrix multiplication: [1, 5, hidden_size] × [1, hidden_size, 5] = [1, 5, 5]
        #
        # What this means:
        # - Each temperature (5 of them) asks: "How well do I match with each temperature (5 of them)?"
        # - Result is a 5×5 matrix where entry [i,j] = how much temperature i should attend to temperature j
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))  # [1, 5, 5]

        # SCALING: Why do we divide by square root of hidden_size?
        # When hidden_size is large, the dot products (queries × keys) can become very large
        # Large values make softmax "sharp" - it puts almost all weight on one element
        # Scaling keeps the values in a reasonable range so attention is more balanced
        scale = self.hidden_size**0.5  # ** 0.5 means square root
        attention_scores = attention_scores / scale

        # F.softmax() converts raw scores into probabilities
        # softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
        # Properties of softmax:
        # 1. All outputs are positive (between 0 and 1)
        # 2. All outputs sum to 1 (like probabilities)
        # 3. Larger input values get exponentially larger output values
        #
        # dim=-1 means apply softmax along the last dimension
        # For each temperature i, softmax is applied across its attention to all 5 temperatures
        attention_weights = F.softmax(attention_scores, dim=-1)  # [1, 5, 5]

        # STEP 5: APPLY ATTENTION TO VALUES 💡
        # Use the attention weights to create a weighted combination of values
        #
        # attention_weights shape: [1, 5, 5]
        # values shape: [1, 5, hidden_size]
        #
        # Matrix multiplication: [1, 5, 5] × [1, 5, hidden_size] = [1, 5, hidden_size]
        #
        # What happens:
        # For each temperature position i:
        #   attended_features[i] = sum over j of (attention_weights[i,j] * values[j])
        #
        # In plain English:
        # "For each temperature, create a new representation that is a weighted average
        #  of ALL temperature values, where the weights come from attention"
        attended_features = torch.matmul(attention_weights, values)  # [1, 5, hidden_size]

        # What just happened is the KEY INSIGHT of attention:
        # - Before: each position only had information about itself
        # - After: each position has information from ALL positions, weighted by importance
        # - The network LEARNED what was important through the attention mechanism!

        # STEP 6: AGGREGATE AND PREDICT 🎯
        # We need to turn our 5 attended features into 1 prediction
        #
        # attended_features has shape [1, 5, hidden_size]
        # We need to reduce this to [1, hidden_size] to make a single prediction
        #
        # .mean(dim=1) calculates the average along dimension 1 (the sequence dimension)
        # This takes the 5 feature vectors and averages them element-wise:
        # final_features[i] = (attended_features[0][i] + attended_features[1][i] + ... + attended_features[4][i]) / 5
        final_features = attended_features.mean(dim=1)  # [1, hidden_size]

        # Alternative aggregation methods we could use:
        # 1. Take only the last position: attended_features[:, -1, :]
        # 2. Take only the first position: attended_features[:, 0, :]
        # 3. Take maximum: attended_features.max(dim=1)[0]
        # 4. Weighted sum with learned weights
        #
        # We chose mean because it's simple and uses information from all positions

        # Make final prediction using a linear layer
        # self.output_layer is nn.Linear(hidden_size, output_size)
        # This transforms [1, hidden_size] → [1, output_size]
        output = self.output_layer(final_features)  # [1, output_size]

        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        loss = F.mse_loss(prediction, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


if __name__ == "__main__":
    example = torch.tensor([[[18.0], [22.0], [15.0], [20.0], [21.0]]])

    model = SimpleAttentionNetwork(input_size=1, sequence_length=5, hidden_size=8, output_size=1)  # Small for demo

    with torch.no_grad():
        result = model(example)
