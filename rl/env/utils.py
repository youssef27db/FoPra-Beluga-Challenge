import torch
from torch.utils.data import Dataset


class Step:
    def __init__(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, old_prob: torch.Tensor, done=torch.tensor(0.0,dtype=torch.float32)):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.old_prob = old_prob
        self.done = done


class StepDataset(Dataset):
    def __init__(self, steps=None):
        """
        Initialisiert das Dataset.
        steps: Liste von Step-Objekten oder Tensors. Optional.
        """
        self.steps = steps if steps is not None else []

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):
        step = self.steps[idx]
        return {
            "state": step.state,
            "action": step.action,
            "reward": step.reward,
            "next_state": step.next_state,
            "old_prob": step.old_prob,
            "done": step.done
        }