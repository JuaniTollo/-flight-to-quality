import torch
import torch.nn as nn
import torch.nn.functional as F

class DialecticalLoss(nn.Module):
    """
    Composite Loss Function for Dialectical Neural Networks (DINN).
    
    Implements Eq. 5.3 from the paper:
        L_total = L_MSE + lambda_1 * L_Orthogonal + lambda_2 * L_Accumulation
    
    Attributes:
        lambda_orth (float): Weight for the orthogonal regularization term (unity vs. struggle).
        lambda_acc (float): Weight for the accumulation penalty (quantity to quality).
    """
    def __init__(self, lambda_orth: float = 0.1, lambda_accumulation: float = 0.01):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_orth = lambda_orth
        self.lambda_acc = lambda_accumulation

    def orthogonal_regularization(self, h_thesis: torch.Tensor, h_antithesis: torch.Tensor) -> torch.Tensor:
        """
        Enforces orthogonality between Thesis and Antithesis latent representations.
        Mathematically minimizes the squared cosine similarity to ensure feature disentanglement.
        
        Args:
            h_thesis: Latent vector of the momentum branch (Batch, Dim).
            h_antithesis: Latent vector of the structural risk branch (Batch, Dim).
            
        Returns:
            Scalar tensor representing the orthogonality penalty.
        """
        # L2 Normalization to project onto the hypersphere
        h_t_norm = F.normalize(h_thesis, p=2, dim=1)
        h_a_norm = F.normalize(h_antithesis, p=2, dim=1)
        
        # Cosine similarity
        cosine = torch.sum(h_t_norm * h_a_norm, dim=1)
        
        # Minimizing squared cosine pushes vectors to be orthogonal (90 degrees)
        return torch.mean(cosine ** 2)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                h_thesis: torch.Tensor, h_antithesis: torch.Tensor) -> torch.Tensor:
        
        # 1. Prediction Error (Standard Reconstruction)
        loss_pred = self.mse(pred, target)
        
        # 2. Dialectical Tension (Regularization)
        loss_orth = self.orthogonal_regularization(h_thesis, h_antithesis)
        
        # Total Loss
        return loss_pred + (self.lambda_orth * loss_orth)