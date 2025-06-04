import torch
from rl.agent.base_agent import BaseAgent

class PPOAgent(BaseAgent):
    def __init__(self, actor, critic, optimizer, clip_epsilon=0.2, gamma=0.99):
        """
        actor: PyTorch-Modell für die Policy
        critic: PyTorch-Modell für den Wertfunktionsschätzer
        optimizer: Optimizer (z.B. Adam) für Actor und Critic
        clip_epsilon: PPO-Clip-Parameter
        gamma: Diskontfaktor
        """
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma

    def select_action(self, state):
        """
        Aktion auswählen basierend auf dem aktuellen Zustand.
        Hier solltest du dein actor Modell nutzen.
        """
        with torch.no_grad():
            action_probs = self.actor(state)
            action = torch.multinomial(action_probs, num_samples=1)
        return action.item()

    def update(self, batch):
        """
        Update-Logik für PPO.
        `batch` ist z.B. ein Batch von Step-Objekten oder Tensoren.
        Implementierung folgt später.
        """
        pass

    def save(self, filepath):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
