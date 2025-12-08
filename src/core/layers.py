import torch
import torch.nn as nn
from typing import Tuple

class AufhebenLayer(nn.Module):
    """
    Dialectical Synthesis Layer (The 'Aufheben' Mechanism).
    
    Unlike standard concatenation, this layer uses a bilinear interaction 
    modulated by a dynamic gating mechanism to model the 'Negation of the Negation'.
    It allows the Antithesis to transform (sublate) the Thesis based on regime context.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        # Projections for dimensionality reduction before synthesis
        self.proj_thesis = nn.Linear(input_dim, output_dim)
        self.proj_antithesis = nn.Linear(input_dim, output_dim)
        
        # Regime Gating Mechanism
        # w -> 1: Thesis dominates (Normal regime, continuity)
        # w -> 0: Antithesis dominates (Crisis regime, rupture)
        self.gate = nn.Sequential(
            nn.Linear(output_dim * 2, 1),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, z_thesis: torch.Tensor, z_antithesis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_thesis: Latent state of the short-term momentum branch.
            z_antithesis: Latent state of the long-term structural branch.
        """
        t = self.proj_thesis(z_thesis)
        a = self.proj_antithesis(z_antithesis)
        
        # Synthesis: Interpenetration of opposites (Hadamard Product)
        # This captures non-linear interaction effects.
        synthesis = t * a 
        
        # Dynamic Gating
        combined = torch.cat([t, a], dim=-1)
        w = self.gate(combined)
        
        # Residual Output (Aufheben)
        # Preserves the identity (t) but modified/negated by the synthesis depending on w.
        out = (w * t) + ((1 - w) * synthesis)
        
        return self.norm(out)