import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math


class SelfAttentionNetwork(pl.LightningModule):
    """
    🚀 SELF-ATTENTION: The Heart of Modern AI

    EVOLUTION: From Simple Attention to Self-Attention

    In our previous attention network:
    - We had separate Query, Key, and Value transformations
    - But they all came from the SAME input features
    - This is actually called "SELF-attention" because the input attends to ITSELF

    What's new in this version:
    1. 🎭 MULTI-HEAD ATTENTION: Multiple attention mechanisms running in parallel
    2. 🔧 PROPER SCALING: Better mathematical foundations
    3. 🏗️ RESIDUAL CONNECTIONS: Skip connections like in ResNet
    4. 🧠 LAYER NORMALIZATION: Stabilizes training
    5. 📚 POSITIONAL ENCODING: Better way to handle positions

    This is essentially a simplified Transformer block!
    """

    def __init__(self, input_size=1, d_model=64, num_heads=4, sequence_length=5, output_size=1, learning_rate=0.01):
        super().__init__()

        # IMPORTANT PARAMETERS:
        self.input_size = input_size  # Size of each input feature (1 for temperature)
        self.d_model = d_model  # Size of our internal representations (like hidden_size but more standard name)
        self.num_heads = num_heads  # How many attention heads to use in parallel
        self.sequence_length = sequence_length

        # REQUIREMENT: d_model must be divisible by num_heads
        # Each head will process d_model // num_heads dimensions
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.head_dim = d_model // num_heads  # Dimension per attention head

        # STEP 1: Input embedding - convert raw input to d_model dimensions
        # This replaces our old input_processor
        self.input_embedding = nn.Linear(input_size, d_model)

        # STEP 2: Positional encoding (learnable parameters)
        # Instead of just adding position numbers, we learn how to represent positions
        self.positional_encoding = nn.Parameter(torch.randn(sequence_length, d_model))

        # STEP 3: Multi-head self-attention
        # These create Q, K, V for ALL heads at once (more efficient than separate layers)
        self.query_projection = nn.Linear(d_model, d_model)  # Projects to all heads' queries
        self.key_projection = nn.Linear(d_model, d_model)  # Projects to all heads' keys
        self.value_projection = nn.Linear(d_model, d_model)  # Projects to all heads' values

        # STEP 4: Output projection for multi-head attention
        # After we get results from all heads, we need to combine them
        self.output_projection = nn.Linear(d_model, d_model)

        # STEP 5: Layer normalization (stabilizes training)
        # This normalizes the activations to have mean=0, std=1
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        # STEP 6: Feed-forward network (like a small MLP)
        # This processes the attended features further
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),  # Expand to 4x size (common practice)
            nn.ReLU(),  # Non-linearity
            nn.Linear(d_model * 4, d_model),  # Contract back to original size
        )

        # STEP 7: Final output layer
        self.output_layer = nn.Linear(d_model, output_size)

        self.learning_rate = learning_rate

    def forward(self, x):
        """
        🔍 Let's trace through self-attention step by step!

        INPUT: x shape [batch_size, sequence_length, input_size]
        Example: [1, 5, 1] for 5 temperatures
        """

        batch_size, seq_len, input_size = x.shape

        # STEP 1: Input embedding
        # Convert each temperature from input_size to d_model dimensions
        # [batch_size, seq_len, input_size] → [batch_size, seq_len, d_model]
        embedded = self.input_embedding(x)  # [1, 5, 64]

        # STEP 2: Add positional encoding
        # self.positional_encoding has shape [seq_len, d_model] = [5, 64]
        # We add this to each example in the batch
        # Broadcasting automatically handles the batch dimension
        positioned = embedded + self.positional_encoding[:seq_len]  # [1, 5, 64]

        # STEP 3: Multi-head self-attention
        attended = self.multi_head_attention(positioned)

        # STEP 4: Residual connection + layer normalization
        # Residual connection: add the input back to the output
        # This helps with training deep networks (from ResNet paper)
        # Layer norm: normalize to mean=0, std=1 for stable training
        norm1 = self.layer_norm1(positioned + attended)  # [1, 5, 64]

        # STEP 5: Feed-forward network
        ff_output = self.feed_forward(norm1)  # [1, 5, 64]

        # STEP 6: Another residual connection + layer normalization
        norm2 = self.layer_norm2(norm1 + ff_output)  # [1, 5, 64]

        # STEP 7: Aggregate sequence information
        # We need to convert [batch_size, seq_len, d_model] to [batch_size, d_model]
        # Using mean pooling (average across sequence dimension)
        pooled = norm2.mean(dim=1)  # [1, 64]

        # STEP 8: Final prediction
        output = self.output_layer(pooled)  # [1, output_size]

        return output

    def multi_head_attention(self, x):
        """
        🎭 Multi-head attention: The crown jewel of transformers!

        Why multiple heads?
        - Different heads can focus on different types of relationships
        - Head 1 might focus on recent trends
        - Head 2 might focus on periodic patterns
        - Head 3 might focus on anomalies
        - etc.

        INPUT: x shape [batch_size, seq_len, d_model]
        """

        batch_size, seq_len, d_model = x.shape

        # STEP 1: Create Q, K, V for all heads at once
        # Each projection creates [batch_size, seq_len, d_model] outputs
        # But we'll reshape them to separate the heads

        queries = self.query_projection(x)  # [1, 5, 64]
        keys = self.key_projection(x)  # [1, 5, 64]
        values = self.value_projection(x)  # [1, 5, 64]

        # STEP 2: Reshape for multi-head processing
        # We want to split d_model into num_heads pieces
        # [batch_size, seq_len, d_model] → [batch_size, seq_len, num_heads, head_dim]
        # Then transpose to [batch_size, num_heads, seq_len, head_dim]

        def reshape_for_heads(tensor):
            """Helper function to reshape tensors for multi-head attention"""
            # tensor shape: [batch_size, seq_len, d_model]
            reshaped = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
            # reshaped: [batch_size, seq_len, num_heads, head_dim]

            # Transpose to put heads dimension first: [batch_size, num_heads, seq_len, head_dim]
            return reshaped.transpose(1, 2)

        q = reshape_for_heads(queries)  # [1, 4, 5, 16]
        k = reshape_for_heads(keys)  # [1, 4, 5, 16]
        v = reshape_for_heads(values)  # [1, 4, 5, 16]

        # STEP 3: Scaled dot-product attention for each head
        # This is the same attention mechanism as before, but applied to each head separately

        # Calculate attention scores: Q × K^T
        # q: [1, 4, 5, 16], k.transpose(-2, -1): [1, 4, 16, 5]
        # Result: [1, 4, 5, 5] - attention scores for each head
        attention_scores = torch.matmul(q, k.transpose(-2, -1))

        # Scale by sqrt(head_dim) to prevent softmax saturation
        # When head_dim is large, dot products can be large, making softmax too sharp
        scale = math.sqrt(self.head_dim)  # sqrt(16) = 4.0
        attention_scores = attention_scores / scale

        # Apply softmax to get attention weights (probabilities)
        attention_weights = F.softmax(attention_scores, dim=-1)  # [1, 4, 5, 5]

        # STEP 4: Apply attention to values
        # attention_weights: [1, 4, 5, 5], v: [1, 4, 5, 16]
        # Result: [1, 4, 5, 16] - attended values for each head
        attended_values = torch.matmul(attention_weights, v)

        # STEP 5: Concatenate heads back together
        # We need to go from [batch_size, num_heads, seq_len, head_dim]
        # back to [batch_size, seq_len, d_model]

        # First transpose: [1, 4, 5, 16] → [1, 5, 4, 16]
        attended_transposed = attended_values.transpose(1, 2)

        # Then reshape: [1, 5, 4, 16] → [1, 5, 64]
        # This concatenates all head outputs: [head0_output, head1_output, head2_output, head3_output]
        attended_concat = attended_transposed.contiguous().view(batch_size, seq_len, d_model)

        # STEP 6: Final linear projection
        # This mixes information from different heads
        output = self.output_projection(attended_concat)  # [1, 5, 64]

        return output

    def training_step(self, batch, batch_idx):
        """Training step - same as before"""
        x, y = batch
        prediction = self(x)
        loss = F.mse_loss(prediction, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        """Optimizer - same as before"""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def create_complex_sequence_data():
    """
    Create data that requires sophisticated attention patterns

    Task: Multi-pattern temperature prediction
    Pattern 1: Cyclical - every 3rd day repeats
    Pattern 2: Trend-based - depends on overall direction
    Pattern 3: Anomaly-based - sudden changes trigger specific responses
    Pattern 4: Long-range - day 0 influences day 4 outcome

    This requires multiple types of attention to solve!
    """
    print("🎲 Creating complex multi-pattern data...")

    sequences = []
    labels = []

    for _ in range(2000):  # More data for complex patterns
        base_temp = torch.rand(1) * 20 + 10  # Base 10-30°C

        # Generate 5 temperatures with multiple overlapping patterns
        temps = torch.zeros(5)

        # Pattern 1: Cyclical component
        for i in range(5):
            cyclical = 3 * math.sin(2 * math.pi * i / 3)  # 3-day cycle
            temps[i] = base_temp + cyclical

        # Pattern 2: Add trend
        trend_strength = (torch.rand(1) - 0.5) * 4  # -2 to +2
        trend = torch.linspace(0, trend_strength, 5)
        temps += trend

        # Pattern 3: Add noise
        noise = torch.randn(5) * 1.5
        temps += noise

        # Pattern 4: Random anomaly
        if torch.rand(1) > 0.7:  # 30% chance
            anomaly_day = torch.randint(0, 5, (1,)).item()
            temps[anomaly_day] += (torch.rand(1) - 0.5) * 8  # Big change

        # COMPLEX PREDICTION RULE:
        # Future temperature depends on:
        # 1. Cyclical pattern continuation
        # 2. Trend strength
        # 3. Anomaly recovery
        # 4. Long-range dependency (day 0 → day 5)

        next_cyclical = 3 * math.sin(2 * math.pi * 5 / 3)
        next_trend = base_temp + trend_strength
        anomaly_effect = (temps - (base_temp + torch.linspace(0, trend_strength, 5))).abs().max() * 0.3
        long_range_effect = (temps[0] - base_temp) * 0.2

        tomorrow = base_temp + next_cyclical + next_trend + torch.randn(1) * 1.0 - anomaly_effect + long_range_effect

        sequences.append(temps.unsqueeze(-1))  # Add feature dimension
        labels.append(tomorrow)

    sequences = torch.stack(sequences)  # [2000, 5, 1]
    labels = torch.stack(labels)  # [2000, 1]

    print(f"✅ Created {len(sequences)} complex sequences")
    print(f"📊 Each sequence has cyclical, trend, anomaly, and long-range patterns")

    return sequences, labels


if __name__ == "__main__":
    print("=" * 90)
    print("🚀 SELF-ATTENTION NETWORK - THE HEART OF MODERN AI")
    print("=" * 90)

    # Create a test example with interesting patterns
    # Temperatures: [15.0, 18.0, 14.0, 17.0, 13.0]
    # Pattern: Alternating up-down with slight downward trend
    example = torch.tensor([[[15.0], [18.0], [14.0], [17.0], [13.0]]])
    print(f"🌡️ Test sequence: {example.squeeze().tolist()}")
    print(f"📈 Pattern: Alternating with downward trend")

    # Create the self-attention network
    model = SelfAttentionNetwork(
        input_size=1,  # Temperature values
        d_model=64,  # Internal representation size
        num_heads=4,  # 4 attention heads
        sequence_length=5,  # 5 time steps
        output_size=1,  # Predict 1 value
    )

    # Process the example
    print("\n" + "=" * 70)
    print("🔄 PROCESSING THROUGH SELF-ATTENTION NETWORK:")
    print("=" * 70)

    with torch.no_grad():  # No gradients needed for demo
        result = model(example)

    print(f"\n🎯 FINAL PREDICTION: {result.item():.2f}°C")

    print("\n" + "=" * 90)
    print("🧠 WHAT YOU JUST LEARNED - THE FOUNDATIONS OF MODERN AI:")
    print("=" * 90)
    print("🎭 MULTI-HEAD ATTENTION:")
    print("   • Multiple attention mechanisms running in parallel")
    print("   • Each head can focus on different types of patterns")
    print("   • Heads are combined to create rich representations")
    print()
    print("🔗 RESIDUAL CONNECTIONS:")
    print("   • Add input back to output: output = layer(input) + input")
    print("   • Helps information flow through deep networks")
    print("   • Prevents vanishing gradients")
    print()
    print("🧪 LAYER NORMALIZATION:")
    print("   • Normalizes activations to mean=0, std=1")
    print("   • Stabilizes training of deep networks")
    print("   • Applied after each major component")
    print()
    print("📍 LEARNABLE POSITIONAL ENCODING:")
    print("   • Network learns how to represent positions")
    print("   • More flexible than fixed position numbers")
    print("   • Can capture complex positional relationships")
    print()
    print("🏗️ TRANSFORMER ARCHITECTURE:")
    print("   • Self-attention + feed-forward + residuals + normalization")
    print("   • This is the basic building block of GPT, BERT, etc.")
    print("   • Stack many of these blocks for powerful models")
    print()
    print("🎉 CONGRATULATIONS! You now understand the core of:")
    print("   • GPT (Generative Pre-trained Transformer)")
    print("   • BERT (Bidirectional Encoder Representations from Transformers)")
    print("   • ChatGPT, Claude, and other modern AI systems")
    print("   • The attention mechanism that revolutionized AI!")
