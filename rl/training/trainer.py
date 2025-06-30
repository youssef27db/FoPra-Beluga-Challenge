import numpy as np
from rl.agents.low_level.heuristics import decide_parameters  # Low-Level-Heuristik
from collections import deque
from mcts import MCTS  # Wenn wir dann MCTS implementiert haben

class Trainer:
    def __init__(self, env, ppo_agent, mcts_params=None):
        self.env = env
        self.ppo_agent = ppo_agent  # High-Level-Agent
        self.mcts = None
        
        # Tracking-Metriken
        self.episode_rewards = []
        self.avg_rewards = []
        self.steps_per_episode = []

    def train(self, n_episodes=1000, save_interval=100):
        for episode in range(n_episodes):
            state = self.env.reset()
            obs = self.env.get_observation(state)
            done = False
            total_reward = 0
            steps = 0

            while not done:
                # High-Level-Entscheidung (PPO)
                high_level_action, _, _ = self.ppo_agent.choose_action(state)
                
                # Low-Level- oder MCTS-Ausführung
                if high_level_action == "HEURISTIC":
                    # Heuristik entscheidet Low-Level-Aktionen
                    action_name, params = decide_parameters(obs, high_level_action)
                elif high_level_action == "MCTS":
                    # MCTS sucht nach der besten Aktion
                    action_name, params = self.mcts.search(state)

                # Führe Aktion aus
                next_state, reward, done, _ = self.env.step(action_name, params)
                
                # Speichere Erfahrung für PPO
                self.ppo_agent.remember(state, high_level_action, _, _, reward, done)
                
                state = next_state
                total_reward += reward
                steps += 1

            # PPO-Lernschritt am Ende der Episode
            self.ppo_agent.learn()
            
            # Metriken speichern
            self.episode_rewards.append(total_reward)
            avg_reward = np.mean(self.episode_rewards[-100:])
            self.avg_rewards.append(avg_reward)
            self.steps_per_episode.append(steps)

            # Fortschritt anzeigen
            if (episode + 1) % save_interval == 0:
                print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Steps: {steps}")
                self.ppo_agent.save_models()

    def evaluate(self, n_episodes=10):
        state = self.env.reset()
        obs = self.env.get_observation(state)
        high_level_action, _, _ = self.ppo_agent.choose_action(state)

        for episode in range(n_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                if np.random.rand() < 0.6:  # 60% Heuristik, 40% MCTS 
                    action_name, params = decide_parameters(obs, high_level_action)
                else:
                    action_name, params = self.mcts.search(state)
                
                next_state, reward, done, _ = self.env.step(action_name, params)
                total_reward += reward
                state = next_state

            print(f"Evaluation Episode {episode + 1}, Reward: {total_reward:.2f}")