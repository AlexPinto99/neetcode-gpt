import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        with torch.no_grad():
            current_input = x
            dead_fractions = []
            for i, layer in enumerate(model):
                total_neurons = current_input.shape[1]
                dead_neurons = 0
                current_output = layer(current_input)
                if isinstance(layer, nn.ReLU):
                    dead_neurons = (current_output == 0).all(dim=0).sum()
                    dead_fractions.append(torch.round(dead_neurons/total_neurons, decimals = 4))
                current_input = current_output
        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        for fraction in dead_fractions:
            if fraction>0.5:
                return 'use_leaky_relu'
        if dead_fractions[0] > 0.3:
            return 'reinitialize'
        increasing = all(dead_fractions[i+1]>dead_fractions[i] for i in range(len(dead_fractions)-1))
        if increasing and dead_fractions[-1]>0.1: return 'reduce_learning_rate'
        return 'healthy'
