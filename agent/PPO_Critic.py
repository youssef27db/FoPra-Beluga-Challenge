import torch
from torch import nn
from torch.nn import functional as F



class PPO_Critic(nn.Module):
    def __init__(self):
      "initialize the critic network for PPO"
      super(PPO_Critic, self).__init__()
      self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
    
    def forward(self, x):
      pass
      # returns value