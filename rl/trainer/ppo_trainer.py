from rl.trainer.base_trainer import BaseTrainer
from rl.data.utils import Step, StepDataset
from torch.utils.data import DataLoader
from rl.agent.ppo.ppo_agent import PPOAgent
import torch


class PPOTrainer(BaseTrainer):
    def __init__(self, env, agent, episodes,clip_epsilon=0.2, gamma=0.99):
        super().__init__(env, agent, episodes)
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma




        def train(self, episodes=1000, update_interval=10, batch_size=64):
            steps = []  # Liste für Step-Objekte
            
            for episode in range(episodes):
                state = self.env.reset()
                done = False
                total_reward = 0
                
                while not done:
                    action, old_prob = self.agent.select_action(state)  # old_prob z.B. Policy Wahrscheinlichkeit
                    next_state, reward, done, _ = self.env.step(action.item())
                    total_reward += reward
                    
                    # Step Objekt erstellen (torch.Tensor nötig)
                    step = Step(
                        state=torch.tensor(state, dtype=torch.float32),
                        action=torch.tensor(action, dtype=torch.float32),
                        reward=torch.tensor([reward], dtype=torch.float32),
                        next_state=torch.tensor(next_state, dtype=torch.float32),
                        old_prob=torch.tensor([old_prob], dtype=torch.float32),
                        done=torch.tensor([float(done)], dtype=torch.float32)
                    )
                    
                    steps.append(step)
                    state = next_state
                
                print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")
                
                # Update alle update_interval Episoden
                if (episode + 1) % update_interval == 0:
                    dataset = StepDataset(steps)
                    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                    self.agent.update(dataloader)  # Agent erwartet jetzt einen DataLoader
                    steps = []  # Buffer leeren
            
            # Optional: am Ende noch einmal updaten
            if steps:
                dataset = StepDataset(steps)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                self.agent.update(dataloader)
            
            print("Training abgeschlossen.")