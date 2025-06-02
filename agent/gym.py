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

    def save(self, filepath="dataset.pth"):
        """
        Speichert das Dataset in eine Datei.
        Die Daten werden als Tensoren gespeichert.
        """
        # Speichern der Schritte als Listen von Tensoren
        states = [step.state for step in self.steps]
        actions = [step.action for step in self.steps]
        rewards = [step.reward for step in self.steps]
        next_states = [step.next_state for step in self.steps]
        old_probs = [step.old_prob for step in self.steps]
        dones = [step.done for step in self.steps]

        # Speichern der Listen als Tensors
        torch.save({
            'states': torch.stack(states),
            'actions': torch.stack(actions),
            'rewards': torch.stack(rewards),
            'next_states': torch.stack(next_states),
            'old_probs': torch.stack(old_probs),
            'dones': torch.stack(dones)
        }, filepath)

        print(f"Dataset erfolgreich unter {filepath} gespeichert.")

    @classmethod
    def load(cls, filepath="dataset.pth"):
        """
        Lädt ein Dataset aus einer Datei, das als Tensors gespeichert wurde.
        """
        # Laden der gespeicherten Tensoren
        data = torch.load(filepath)

        # Umwandeln der geladenen Tensoren in Step-Objekte
        steps = [
            Step(state=item[0], action=item[1], reward=item[2], next_state=item[3], old_prob=item[4].detach(),done=item[5])
            for item in zip(data['states'], data['actions'], data['rewards'], data['next_states'], data['old_probs'], data['dones'])
        ]

        print(f"Dataset erfolgreich aus {filepath} geladen.")
