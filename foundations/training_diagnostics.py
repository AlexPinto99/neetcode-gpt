import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        layers = []
        with torch.no_grad():
            current_input = x
            for i, layer in enumerate(model):
                current_output = layer(current_input)
                if isinstance(layer,nn.Linear):
                    layer_dict={}
                    layer_dict["mean"] = round(current_output.detach().mean().item(), 4)
                    layer_dict["std"] = round(current_output.detach().std().item(), 4)
                    layer_dict["dead_fraction"] = round((current_output <= 0).all(dim=0).float().detach().mean().item(), 4)
                    layers.append(layer_dict)
                current_input=current_output
        return layers

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        stats = []
        model.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        for i, layer in enumerate(model):
            if isinstance(layer,nn.Linear):
                grad = layer.weight.grad
                if grad is not None:
                    layer_dict={}
                    layer_dict["mean"] = round(grad.detach().mean().item(), 4)
                    layer_dict["std"] = round(grad.detach().std().item(), 4)
                    layer_dict["norm"] = round(torch.norm(grad).item(), 4)
                    stats.append(layer_dict)
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        for stat in activation_stats:
            if stat["dead_fraction"]>0.5: return "dead_neurons"
        for stat in gradient_stats:
            if stat["norm"] > 1000: return "exploding_gradients"
        if gradient_stats[-1]["norm"] < 1e-5: return "vanishing_gradients"
        for stat in activation_stats:
            if stat["std"] <0.1: return "vanishing_gradients"
            if stat["std"]>10.0: return "exploding_gradients"
        return "healthy"
