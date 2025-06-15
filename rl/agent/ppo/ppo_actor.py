import torch
import torch.nn as nn
import torch.nn.functional as F



class ActorCNNV1(nn.Module):
    def __init__(self):
        super(ActorCNNV1, self).__init__()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        pass
    
    def forward(self, x) -> torch.Tensor:
        """
        returns the action probabilities for the given input state.
        """
        pass
