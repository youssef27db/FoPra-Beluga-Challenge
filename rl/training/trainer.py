import numpy as np
from rl.env.environment import * # Environment 
from rl.agents.high_level.ppo_agent import *  # High-Level-Agent
from rl.agents.low_level.heuristics import *  # Low-Level-Heuristik
from rl.mcts import *  # MCTS-Algorithmus

class Trainer:
    def __init__(self, env: Env, ppo_agent: PPOAgent, mcts_params=None):
        self.env = env
        self.ppo_agent: PPOAgent = ppo_agent  # High-Level-Agent
        self.mcts = None
        
        # Tracking-Metriken
        self.episode_rewards = []
        self.avg_rewards = []
        self.steps_per_episode = []
        self.best_score = 1
        self.score_history = []
        self.learn_iters = 0

        # Number to Actoin Mapping
        self.action_mapping = {
            0 : "load_beluga",
            1 : "unload_beluga",
            2 : "get_from_hangar",
            3 : "deliver_to_hangar",
            4 : "left_stack_rack",
            5 : "right_stack_rack",
            6 : "left_unstack_rack",
            7 : "right_unstack_rack"
        }

    def train(self, n_episodes=1000, N=20):
        for episode in range(n_episodes):
            obs = self.env.reset()
            isTerminal = False
            total_reward = 0
            steps = 0

            while not isTerminal:
                # High-Level-Entscheidung (PPO)
                high_level_action, prob, val = self.ppo_agent.choose_action(obs)
                high_level_action_str = self.action_mapping[high_level_action]  # Mapping der Aktion

                # Low-Level-Agent: 
                # Heuristik
                action_name, params = decide_parameters(obs, high_level_action_str)
                print(f"High-Level-Action: {high_level_action_str}, Heuristic Action: {action_name}, Params: {params}")
                # Wenn keine Heuristik gefunden wurde, dann MCTS verwenden
                if action_name is "None":
                    root = MCTSNode(state=self.env.state, action=(high_level_action_str, None))

                    # MCTS mit diesem Root-Node starten
                    mcts = MCTS(root, depth=10, n_simulations=10)
                    best_node = mcts.search()
                    
                    if best_node:
                        params = best_node.action[1]
                        print("-" *20)
                        print(params)
                        print("-" *20)
                        print(f"Beste Parameter für {high_level_action_str}: {params}")
                
                
                print("-" *20)
                print(params)
                print("-" *20)

                # Führe Aktion aus
                obs_ , reward, isTerminal = self.env.step(high_level_action_str, params)
                
                # Speichere Erfahrung für PPO
                self.ppo_agent.remember(obs, high_level_action, prob, val, reward, isTerminal)

                obs = obs_
                total_reward += reward
                steps += 1

                # PPO-Lernschritt am Ende der Episode
                if steps % N == 0:
                    self.ppo_agent.learn()
                    self.learn_iters += 1

                if steps > 2000:
                    isTerminal = True  # Abbruchbedingung, um zu lange Episoden zu vermeiden
    
            # Metriken speichern
            self.episode_rewards.append(total_reward)
            avg_reward = np.mean(self.episode_rewards[-100:])
            self.avg_rewards.append(avg_reward)
            self.steps_per_episode.append(steps)

            # Fortschritt anzeigen
            if avg_reward > self.best_score:
                self.ppo_agent.save_models()

            print('episode', episode, 'score %.1f' % total_reward, 'avg score %.1f' % avg_reward,
              'time_steps', steps, 'learn_iters', self.learn_iters)

    # def evaluate(self, n_episodes=10):
    #     state = self.env.reset()
    #     obs = self.env.get_observation(state)
    #     high_level_action, _, _ = self.ppo_agent.choose_action(state)

    #     for episode in range(n_episodes):
    #         state = self.env.reset()
    #         done = False
    #         total_reward = 0

    #         while not done:
    #             if np.random.rand() < 0.6:  # 60% Heuristik, 40% MCTS 
    #                 action_name, params = decide_parameters(obs, high_level_action)
    #             else:
    #                 action_name, params = self.mcts.search(state)
                
    #             next_state, reward, done, _ = self.env.step(action_name, params)
    #             total_reward += reward
    #             state = next_state

    #         print(f"Evaluation Episode {episode + 1}, Reward: {total_reward:.2f}")