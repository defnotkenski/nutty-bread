import torch
import torch.nn as nn
import torch.nn.functional as fnn
from torch import Tensor


class CompetitorWithConfidence(nn.Module):
    """
    🧠 ANALOGY: Like a specialist detective who examines evidence and rates their own confidence

    Each competitor is like a detective specialist (e.g., fingerprint expert, DNA analyst).
    They examine the same evidence (input features) but look for different patterns.
    After processing, they rate how confident they are about what they found.

    INPUT: Raw evidence/features (..., d_model) - e.g., horse racing data like odds, track conditions
    OUTPUT: (processed_evidence, confidence_score) - their analysis + how sure they are about it

    EXAMPLE:
    - Competitor 1: "I found strong track condition patterns" → confidence = 0.9
    - Competitor 2: "I found weak historical patterns" → confidence = 0.3
    """

    def __init__(self, in_features):
        super().__init__()
        self.processor = nn.Sequential(nn.Linear(in_features, in_features), nn.GELU())
        self.confidence_head = nn.Linear(in_features, 1)

    def forward(self, x):
        # Detective analyzes the evidence
        activation = self.processor(x)
        # Detective rates their confidence
        confidence = self.confidence_head(activation)
        return activation, confidence.squeeze(-1)


class LWTA(nn.Module):
    """
    🧠 ANALOGY: Like a jury selecting which detective's testimony to trust most

    Multiple detective specialists examine the same evidence simultaneously.
    Each detective processes the evidence and rates their confidence.
    The jury (gating mechanism) decides which detective to listen to based on confidence.
    During training: "Let's hear from multiple detectives" (soft selection)
    During inference: "Only listen to the most confident detective" (hard selection)

    INPUT: Evidence to analyze (..., d_model)
    OUTPUT: The chosen detective's processed evidence (..., d_model)

    FLOW:
    1. All detectives analyze the evidence → (activations, confidences)
    2. Jury compares confidence scores → gates
    3. Select winning detective's analysis → final output
    """

    def __init__(self, in_features: int, num_competitors: int, temp: float):
        super().__init__()
        self.in_features = in_features
        self.num_competitors = num_competitors
        self.temp = temp

        self.competitors = nn.ModuleList([CompetitorWithConfidence(in_features) for _ in range(num_competitors)])

    def forward(self, x: Tensor):
        """
        🔍 PROCESS: "Let all detectives examine the evidence, then pick the best analysis"

        1. All competitors process input → get (activation, confidence) pairs
        2. Compare confidence scores to decide winner(s)
        3. Output the weighted combination of winners' analyses
        """
        assert x.size(-1) == self.in_features, f"Expected {self.in_features} features, got {x.size(-1)}"

        # All detectives examine the same evidence
        competitor_outputs = [comp(x) for comp in self.competitors]

        # Collect detective findings and confidence scores
        activations = torch.stack([output[0] for output in competitor_outputs], dim=-1)
        competition_scores = torch.stack([output[1] for output in competitor_outputs], dim=-1)

        if self.training:
            # Let's hear from mulitple detectives
            gates = fnn.gumbel_softmax(competition_scores, tau=self.temp, hard=False, dim=-1)
        else:
            # Only listen to the most confidence detective
            gates = fnn.one_hot(competition_scores.argmax(dim=-1), num_classes=self.num_competitors).float()

        # Make gates match the shape of detective findings
        gates_expanded = gates.unsqueeze(-2)
        # Apply jury's trust levels to each detective's testimony
        weighted_acts = activations * gates_expanded
        # Blend all weighted testimonies into final verdict
        output = weighted_acts.sum(dim=-1)

        return output

    def update_temperature(self, new_temp: float):
        self.temp = new_temp
