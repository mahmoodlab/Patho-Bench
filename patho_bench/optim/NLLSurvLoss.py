import torch
import torch.nn as nn

"""
Refactored from https://github.com/mahmoodlab/PORPOISE/blob/master/utils/loss_func.py
"""

class NLLSurvLoss(nn.Module):
    """
    The negative log-likelihood loss function for the discrete time-to-event model (Zadeh and Schmid, 2020).
    Code originally adapted from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py

    Parameters
    ----------
    alpha : float, optional
        Controls the balance between the survival and censoring terms in the loss calculation.
        A higher value of alpha gives more weight to the survival term, while a lower value gives more weight to the censoring term.
        Default is 0.0, which gives equal weight to both terms.
    eps : float, optional
        Numerical constant to avoid taking logs of tiny numbers. Default is 1e-7.
    reduction : str, optional
        Specifies the reduction to apply to the output: 'mean' | 'sum'.
        'mean': the sum of the output will be divided by the number of elements in the output.
        'sum': the output will be summed. Default is 'mean'.
    """
    def __init__(self, alpha=0.0, eps=1e-7, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def __call__(self, x, y_bins, y_event):
        """
        The negative log-likelihood loss function for the discrete time-to-event model (Zadeh and Schmid, 2020).
        Code originally adapted from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (n_batches, n_classes) representing the neural network output logits.
            Hazards are computed as sigmoid(h).
        y_bins : torch.Tensor
            Tensor of shape (n_batches, 1) representing the true time bin index labels. Ranges from 0 to n_classes - 1.
        y_event : torch.Tensor
            Tensor of shape (n_batches, 1) representing the event status indicators.

        Returns
        -------
        torch.Tensor
            The computed negative log-likelihood loss.

        References
        ----------
        Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
        """
        # Convert y_event to y_censor
        y_event = y_event.type(torch.int64)
        y_bins = y_bins.type(torch.int64)

        y_censor = 1 - y_event

        # Compute hazards using sigmoid activation
        hazards = torch.sigmoid(x)

        # Compute survival probabilities
        S = torch.cumprod(1 - hazards, dim=1)

        # Pad survival probabilities with ones at the beginning
        S_padded = torch.cat([torch.ones_like(y_censor), S], 1)

        # Gather previous and current survival probabilities and hazards
        s_prev = torch.gather(S_padded, dim=1, index=y_bins).clamp(min=self.eps)
        h_this = torch.gather(hazards, dim=1, index=y_bins).clamp(min=self.eps)
        s_this = torch.gather(S_padded, dim=1, index=y_bins + 1).clamp(min=self.eps)

        # Compute uncensored and censored loss components
        uncensored_loss = -(1 - y_censor) * (torch.log(s_prev) + torch.log(h_this))
        censored_loss = -y_censor * torch.log(s_this)

        # Combine losses
        loss = uncensored_loss + (1 - self.alpha) * censored_loss

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}. Must be 'mean' or 'sum'.")