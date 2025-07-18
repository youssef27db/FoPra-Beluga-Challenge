import numpy as np
from rl.env.environment import * # Environment 
from rl.agents.high_level.ppo_agent import *  # High-Level-Agent
from rl.agents.low_level.heuristics import *  # Low-Level-Heuristik
from rl.mcts import *  # MCTS-Algorithmus
from rl.utils.utils import *

class Trainer:
    def __init__(self, env: Env, ppo_agent: PPOAgent, mcts_params=None):
        self.env = env
        self.ppo_agent: PPOAgent = ppo_agent  # High-Level-Agent
        self.mcts = None
        
        # Tracking-Metriken
        self.episode_rewards = []
        self.avg_rewards = []
        self.steps_per_episode = []
        self.best_score = -90000
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

    def train(self, n_episodes=2000, N=5, max_steps_per_episode = 200, train_on_old_models = False, start_learn_after = 1000):
        if train_on_old_models:
            self.ppo_agent.load_models()  # Lade die Modelle des PPO-Agenten
        total_steps = 0

        for episode in range(n_episodes):
            obs = self.env.reset()
            isTerminal = False
            total_reward = 0
            steps = 0
            last_trailer_id = None
            last_rack_id = None
            last_action = None
            positive_actions_reward = 0
            print_action = None
            permutation = np.random.permutation(10)

            while not isTerminal:
                obs_ = None  # Reset der Beobachtung für die nächste Iteration
                bool_heuristic = False
                reward = 0
                # High-Level-Entscheidung (PPO)
                permuted_obs = permute_high_level_observation(permutation, obs)
                high_level_action, prob, val, dist = self.ppo_agent.choose_action(permuted_obs)
                high_level_action_str = self.action_mapping[high_level_action]  # Mapping der Aktion

                debuglog("-" *20)
                debuglog(high_level_action_str)
                debuglog("-" *20)

                probs = dist.probs.detach().cpu().numpy()

                while not self.env.check_action_execution(high_level_action_str, obs):
                    if np.all(probs == 0):
                        print("No valid action found, stopping episode.")
                        isTerminal = True
                        reward -= 1000000.0  # Bestrafe Episode, wenn keine Aktionen auführbar sind
                        break

                    probs[high_level_action] = 0.0  # Setze die Wahrscheinlichkeit der aktuellen Aktion auf 0
                    high_level_action = np.argmax(probs)  # Wähle die Aktion mit der höchsten Wahrscheinlichkeit
                    high_level_action_str = self.action_mapping[high_level_action]  # Aktualisiere den Aktionsnamen
                    prob = probs[high_level_action]  # Aktualisiere die Wahrscheinlichkeit der Aktion

                if not isTerminal:
                    # Low-Level-Agent: 
                    # Heuristik
                    action_name, params = decide_parameters(obs, high_level_action_str)
                    # Wenn keine Heuristik gefunden wurde, dann MCTS verwenden
                    if action_name == "None":
                        root = MCTSNode(state=self.env.state, action=(high_level_action_str, None))

                        # MCTS mit diesem Root-Node starten
                        mcts = MCTS(root, depth=10, n_simulations=10)
                        best_node = mcts.search()
                        
                        if best_node:
                            params = best_node.action[1]
                    else:
                        bool_heuristic = True

                    debuglog("-" *20)
                    debuglog(params)
                    debuglog("-" *20)
            

                    # Überprüfe, ob wir eine loop haben mit unstacking und stacking
                    params_check = list(params.values()) if isinstance(params, dict) else list(params)
                    if params_check != []:
                        if (high_level_action == 4 and last_action == 6) or (high_level_action == 5 and last_action == 7) \
                            or (high_level_action == 6 and last_action == 4) or (high_level_action == 7 and last_action == 5):
                            if last_trailer_id == params_check[1] and last_rack_id == params_check[0]:
                                reward -= 1000.0  
                        last_action = high_level_action
                        if last_action in [4, 5, 6, 7]:
                            last_trailer_id = params_check[1]
                            last_rack_id = params_check[0] 


                    # Führe Aktion aus
                    obs_ , reward_main, isTerminal = self.env.step(high_level_action_str, params)
                    reward += reward_main

                    # Wenn besondere Heuristik verwendet wurde, dann führe folge von Aktionen aus: left_unstack -> load_beluga; right_unstack -> deliver_to_hangar
                    if bool_heuristic:
                        if high_level_action_str == "right_unstack_rack":
                            action_name, params = decide_parameters(obs_, "deliver_to_hangar")
                            if not action_name == "None":
                                obs_ , reward_heuristic, isTerminal = self.env.step("deliver_to_hangar", params)
                                reward += reward_heuristic
                        elif high_level_action_str == "left_unstack_rack":
                            action_name, params = decide_parameters(obs_, "load_beluga")
                            if not action_name == "None":
                                obs_ , reward_heuristic, isTerminal = self.env.step("load_beluga", params)
                                reward += reward_heuristic
                        reward += 5.0 # Füge Heuristik-Belohnung hinzu
                    

                print_action = high_level_action_str  # Speichere die letzte Aktion für Debugging
                # Speichere Erfahrung für PPO
                self.ppo_agent.remember(obs, high_level_action, prob, val, reward, isTerminal)

                if reward > 0: 
                    positive_actions_reward += reward
                if not obs_ is None:
                    obs = obs_
                total_reward += reward
                steps += 1
                total_steps += 1

                # PPO-Lernschritt am Ende der Episode
                if total_steps >= start_learn_after and total_steps % N == 0:
                    self.ppo_agent.learn()
                    self.learn_iters += 1

                debuglog(steps)
                if steps >= max_steps_per_episode or total_reward <= -20000:
                    isTerminal = True  # Abbruchbedingung, um zu lange Episoden zu vermeiden
    
            # Metriken speichern
            self.episode_rewards.append(total_reward)
            avg_reward = np.mean(self.episode_rewards[-10:])
            self.avg_rewards.append(avg_reward)
            self.steps_per_episode.append(steps)

            # Fortschritt anzeigen
            if avg_reward > self.best_score:
                self.ppo_agent.save_models()
                self.best_score = avg_reward

            print('episode', episode, 'score %.1f' % total_reward, 'avg score %.1f' % avg_reward, 'Best avg score %.1f' % self.best_score,
              'time_steps', steps, 'learn_iters', self.learn_iters, 'positive reward', positive_actions_reward, 'last_action', print_action)

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