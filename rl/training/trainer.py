"""!
@file trainer.py
@brief Training orchestrator for the Beluga Challenge

This module implements the main training loop that coordinates
the RL agent, MCTS, and environment for the container optimization problem.
"""

import numpy as np
from rl.env.environment import * # Environment 
from rl.agents.high_level.ppo_agent import *  # High-Level-Agent
from rl.agents.low_level.heuristics import *  # Low-Level-Heuristic
from rl.mcts import *  # MCTS-Algorithm
from rl.utils.utils import *
import matplotlib.pyplot as plt

class Trainer:
    """!
    @brief Main training orchestrator for the Beluga Challenge
    
    This class manages the training process, coordinating between
    the RL agent, MCTS, and environment components.
    """
    
    def __init__(self, env: Env, ppo_agent: PPOAgent, mcts_params=None, debug=False):
        """!
        @brief Initialize the trainer
        @param env Environment instance
        @param ppo_agent PPO agent for high-level decisions
        @param mcts_params Parameters for MCTS (optional)
        @param debug Enable debug output
        """
        self.env = env
        self.ppo_agent: PPOAgent = ppo_agent  # High-Level-Agent
        self.mcts = None
        self.debug = debug  # Debug mode for additional output
        
        # Tracking metrics
        self.episode_rewards = []
        self.avg_rewards = []
        self.steps_per_episode = []
        self.best_score = -90000
        self.score_history = []
        self.learn_iters = 0
        self.invalid_action_counts = {i: 0 for i in range(8)}  # Counter for invalid actions by type
        
        # Exploration parameters
        self.epsilon_start = 0.9  # Initial value for epsilon (exploration probability)
        self.epsilon_end = 0.2   # Final value for epsilon
        self.epsilon_decay = 0.00001  # Rate at which epsilon is reduced
        self.total_steps = 0      # Total number of steps taken
        
        # Number to Action Mapping
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
        
    def get_valid_actions(self, obs):
        """!
        @brief Check which actions are valid in the current state
        @param obs Current observation
        @return List of valid action indices
        """
        valid_actions = []
        for action_idx in range(len(self.action_mapping)):
            if self.env.check_action_execution(self.action_mapping[action_idx], obs):
                valid_actions.append(action_idx)
        return valid_actions

    def train(self, n_episodes=2000, N=5, max_steps_per_episode = 200, train_on_old_models = False, start_learn_after = 500, use_permutation = False):
        """!
        @brief Train the agent over a specified number of episodes
        @param n_episodes Number of training episodes
        @param N Frequency of learning steps
        @param max_steps_per_episode Maximum number of steps per episode
        @param train_on_old_models Whether to load existing models
        @param start_learn_after After how many steps learning should begin
        @param use_permutation Whether observations should be permuted (can stabilize training but costs time)
        """
        if train_on_old_models:
            self.ppo_agent.load_models()  # Load the PPO agent's models
        self.total_steps = 0

        for episode in range(n_episodes):
            obs = self.env.reset()
            isTerminal = False
            total_reward = 0
            steps = 0
            last_trailer_id = None
            last_rack_id = None
            last_action = None
            positive_actions_reward = 0

            while not isTerminal:
                bool_heuristic = False
                reward = 0
                # High-Level decision (PPO)
                # Calculate current epsilon value for exploration
                epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                          np.exp(-self.epsilon_decay * self.total_steps)
                
                # Epsilon-Greedy strategy for exploration
                if np.random.random() < epsilon:
                    # Explorative action: Choose a random valid action
                    valid_actions = self.get_valid_actions(obs)
                    if valid_actions:
                        high_level_action = np.random.choice(valid_actions)
                        high_level_action_str = self.action_mapping[high_level_action]
                        
                        # To maintain PPO logic, we need the distribution
                        if use_permutation:
                            _, prob, val, dist = self.ppo_agent.choose_action(permute_high_level_observation(np.random.permutation(10), obs))
                        else:
                            _, prob, val, dist = self.ppo_agent.choose_action(obs)
                    else:
                        # If no valid actions are available, use normal strategies
                        if use_permutation:
                            obs_ = None  # Reset observation for next iteration
                            permutation = np.random.permutation(10)
                            permuted_obs = permute_high_level_observation(permutation, obs)
                            high_level_action, prob, val, dist = self.ppo_agent.choose_action(permuted_obs)
                        else:
                            high_level_action, prob, val, dist = self.ppo_agent.choose_action(obs)
                        high_level_action_str = self.action_mapping[high_level_action]
                else:
                    # Exploitative action: Use PPO policy
                    if use_permutation:
                        obs_ = None  # Reset observation for next iteration
                        permutation = np.random.permutation(10)
                        permuted_obs = permute_high_level_observation(permutation, obs)
                        high_level_action, prob, val, dist = self.ppo_agent.choose_action(permuted_obs)
                    else:
                        high_level_action, prob, val, dist = self.ppo_agent.choose_action(obs)
                    high_level_action_str = self.action_mapping[high_level_action]  # Action mapping

                probs = dist.probs.detach().cpu().numpy()
                
                # Check which actions are valid
                valid_actions = self.get_valid_actions(obs)
                
                # If no valid actions exist, error message and end episode
                if not valid_actions:
                    print(f"Keine gültigen Aktionen für diesen Zustand möglich. Problem: {self.env.problem_name}")
                    isTerminal = True
                    reward -= 5000.0  # Reduced penalty since it's really impossible
                    break
                
                # Balance actions - if too many unstack actions were used,
                # reduce their probability in favor of other actions
                if episode > 100 and self.total_steps > 500:  # After a warm-up phase
                    total_invalid = sum(self.invalid_action_counts.values())
                    if total_invalid > 0:
                        unstack_percentage = (self.invalid_action_counts[6] + self.invalid_action_counts[7]) / total_invalid
                        if unstack_percentage > 0.4:  # If more than 40% of invalid actions are unstacks
                            # Reduce probability for unstack actions
                            unstack_idx = [6, 7]  # left_unstack_rack, right_unstack_rack
                            scale_factor = 0.5  # Scale probability down
                            for idx in unstack_idx:
                                if idx < len(probs):
                                    probs[idx] *= scale_factor
                            # Normalize probabilities again
                            if np.sum(probs) > 0:
                                probs = probs / np.sum(probs)
                                
                            # Update action choice
                            high_level_action = np.argmax(probs)
                            high_level_action_str = self.action_mapping[high_level_action]
                            prob = probs[high_level_action]
                    
                # Debug output for valid actions, if enabled
                if self.debug and steps % 10 == 0:  # Don't output too often
                    valid_action_names = [self.action_mapping[idx] for idx in valid_actions]
                    print(f"Gültige Aktionen: {valid_action_names}")

                # Try at most all actions
                tried_actions = set()
                while not self.env.check_action_execution(high_level_action_str, obs):
                    # Count invalid actions for later analysis
                    self.invalid_action_counts[high_level_action] += 1
                    tried_actions.add(high_level_action)
                    
                    if len(tried_actions) >= len(self.action_mapping):
                        print(f"Alle Aktionen probiert, keine ist gültig. Problem: {self.env.problem_name}")
                        isTerminal = True
                        reward -= 10000.0
                        break

                    probs[high_level_action] = 0.0
                    
                    # If all remaining probabilities are 0, choose randomly from untried actions
                    if np.all(probs == 0):
                        untried_actions = [i for i in range(len(self.action_mapping)) if i not in tried_actions]
                        if untried_actions:
                            high_level_action = np.random.choice(untried_actions)
                        else:
                            print("No valid action found, stopping episode.")
                            isTerminal = True
                            reward -= 10000.0
                            break
                    else:
                        high_level_action = np.argmax(probs)
                    
                    high_level_action_str = self.action_mapping[high_level_action]  
                    prob = probs[high_level_action] 

                if not isTerminal:
                    # Low-Level-Agent: 
                    # Heuristik
                    action_name, params = decide_parameters(obs, high_level_action_str)
                    # If no heuristic found, use MCTS 
                    if action_name == "None":
                        root = MCTSNode(state=self.env.state, action=(high_level_action_str, None))
                        mcts = MCTS(root, depth=5, n_simulations=60)
                        best_node = mcts.search()
                        
                        if best_node:
                            params = best_node.action[1]
                    else:
                        bool_heuristic = True


                    # Check if there is a loop in the actions
                    params_check = list(params.values()) if isinstance(params, dict) else list(params)
                    if params_check != []:
                        if (high_level_action == 4 and last_action == 6) or (high_level_action == 5 and last_action == 7) \
                            or (high_level_action == 6 and last_action == 4) or (high_level_action == 7 and last_action == 5):
                            if last_trailer_id == params_check[1] and last_rack_id == params_check[0]:
                                reward -= 200.0 
                        last_action = high_level_action
                        if last_action in [4, 5, 6, 7]:
                            last_trailer_id = params_check[1]
                            last_rack_id = params_check[0] 


                    # Führe Aktion aus
                    obs_ , reward_main, isTerminal = self.env.step(high_level_action_str, params)
                    reward += reward_main

                    # Wenn besondere Heuristik verwendet wurde, dann führe folge von Aktionen aus: left_unstack -> load_beluga; right_unstack -> deliver_to_hangar
                    if bool_heuristic:
                        # Reduziere die Belohnung für "unstack" Aktionen, wenn die Folgeaktion nicht ausgeführt werden kann
                        if high_level_action_str == "right_unstack_rack":
                            action_name, params = decide_parameters(obs_, "deliver_to_hangar")
                            if not action_name == "None":
                                obs_ , reward_heuristic, isTerminal = self.env.step("deliver_to_hangar", params)
                                reward += reward_heuristic
                                reward += 50.0  # Erhöhte Belohnung für erfolgreiche Aktionskette
                            else:
                                # Bestrafe das Unstacking ohne Folgeaktion
                                reward -= 20.0
                        elif high_level_action_str == "left_unstack_rack":
                            action_name, params = decide_parameters(obs_, "load_beluga")
                            if not action_name == "None":
                                obs_ , reward_heuristic, isTerminal = self.env.step("load_beluga", params)
                                reward += reward_heuristic
                                reward += 50.0  # Erhöhte Belohnung für erfolgreiche Aktionskette
                            else:
                                # Penalize unstacking without follow-up action
                                reward -= 20.0
                        else:
                            # Other heuristics receive smaller rewards
                            reward += 5.0
                    

                print_action = high_level_action_str  # Store last action for debugging
                # Store experience for PPO
                self.ppo_agent.remember(obs, high_level_action, prob, val, reward, isTerminal)

                if reward > 0: 
                    positive_actions_reward += reward
                if not obs_ is None:
                    obs = obs_
                total_reward += reward
                steps += 1
                self.total_steps += 1

                # PPO learning step at end of episode (optimized frequency)
                if self.total_steps >= start_learn_after and self.total_steps % (N*2) == 0:
                    self.ppo_agent.learn()
                    self.learn_iters += 1

                # debuglog(steps) # Debug output disabled
                if steps >= self.env.get_max_steps() or total_reward <= -10000:
                    isTerminal = True  # Adjusted termination condition with less strict reward limit
    
            # Save metrics
            self.episode_rewards.append(total_reward)
            avg_reward = np.mean(self.episode_rewards[-10:])
            self.avg_rewards.append(avg_reward)
            self.steps_per_episode.append(steps)
            
            # Check if epsilon reset is needed
            # If the last 6 episodes all have very bad rewards, reset epsilon
            if len(self.episode_rewards) >= 6:
                recent_rewards = self.episode_rewards[-6:]
                if all(reward <= -10000 for reward in recent_rewards):
                    print("\nSehr schlechte Performance in den letzten 10 Episoden. Setze Epsilon zurück, um mehr zu explorieren.")
                    self.epsilon_start = 0.9  # Zurücksetzen auf Anfangswert
                    self.epsilon_decay = 0.00001  # Zurücksetzen der Decay-Rate
                    self.total_steps = 0  # Zurücksetzen der Schritte für die Epsilon-Berechnung

            # Save model if average reward improves
            if avg_reward > self.best_score:
                self.ppo_agent.save_models()
                self.best_score = avg_reward

            # Check if the problem is solved
            solved = self.env.state.is_terminal()
            status_symbol = "✅" if solved else "  "
            
            print(f'{status_symbol} episode {episode}, score {total_reward:.1f}, avg score {avg_reward:.1f}, Best avg score {self.best_score:.1f}',
                  f'time_steps {steps}/{self.env.get_max_steps()}, learn_iters {self.learn_iters}, positive reward {positive_actions_reward:.1f}, problem {self.env.problem_name}, {self.env.base_index}')
                  
            # Save model every 100 episodes
            if episode > 0 and episode % 100 == 0:
                self.ppo_agent.save_models()



    def evaluateModel(self, n_eval_episodes=10, max_steps_per_episode=200, plot = False):
        """
        @brief Evaluates the model over a specific number of episodes
        @param n_eval_episodes Number of episodes for evaluation (default: 10)
        @param max_steps_per_episode Maximum steps per episode (default: 200)
        @param plot Whether to plot results (default: False)
        @return tuple containing average reward, standard deviation, and steps data
        """
        self.ppo_agent.load_models()
        total_rewards = []
        steps_list = []

        for ep in range(n_eval_episodes):
            obs = self.env.reset()
            isTerminal = False
            total_reward = 0
            steps = 0
            last_trailer_id = None
            last_rack_id = None
            last_action = None

            while not isTerminal and steps < max_steps_per_episode:

                # Choose action without learning
                _, _, _, dist = self.ppo_agent.choose_action(obs)

                probs = dist.probs.detach().cpu().numpy() 
                high_level_action = np.argmax(probs)
                high_level_action_str = self.action_mapping[high_level_action]

                # Check which actions are valid
                valid_actions = self.get_valid_actions(obs)
                
                # If no valid actions exist, end episode
                if not valid_actions:
                    print(f"[Eval] Keine gültigen Aktionen für diesen Zustand. Problem: {self.env.problem_name}")
                    isTerminal = True
                    break

                # Try at most all actions
                tried_actions = set()
                while not self.env.check_action_execution(high_level_action_str, obs):
                    tried_actions.add(high_level_action)
                    
                    if len(tried_actions) >= len(self.action_mapping):
                        print(f"[Eval] Alle Aktionen probiert, keine ist gültig. Problem: {self.env.problem_name}")
                        isTerminal = True
                        break

                    # Set probability of current action to 0
                    probs[high_level_action] = 0.0
                    
                    # If all remaining probabilities are 0
                    if np.all(probs == 0):
                        # Choose randomly from actions not yet tried
                        untried_actions = [i for i in range(len(self.action_mapping)) if i not in tried_actions]
                        if untried_actions:
                            high_level_action = np.random.choice(untried_actions)
                        else:
                            isTerminal = True
                            break
                    else:
                        # Choose action with highest probability
                        high_level_action = np.argmax(probs)
                    
                    high_level_action_str = self.action_mapping[high_level_action]

                if isTerminal:
                    break

                # Low-Level Agent
                action_name, params = decide_parameters(obs, high_level_action_str)
                if action_name == "None":
                    root = MCTSNode(state=self.env.state, action=(high_level_action_str, None))
                    mcts = MCTS(root, depth=3, n_simulations=3)  # Reduced parameters for faster execution
                    best_node = mcts.search()
                    if best_node:
                        params = best_node.action[1]

                # Loop prevention
                params_check = list(params.values()) if isinstance(params, dict) else list(params)
                if params_check != []:
                    if (high_level_action == 4 and last_action == 6) or (high_level_action == 5 and last_action == 7) \
                        or (high_level_action == 6 and last_action == 4) or (high_level_action == 7 and last_action == 5):
                        if last_trailer_id == params_check[1] and last_rack_id == params_check[0]:
                            total_reward -= 1000.0
                    last_action = high_level_action
                    if last_action in [4, 5, 6, 7]:
                        last_trailer_id = params_check[1]
                        last_rack_id = params_check[0]

                obs, reward, isTerminal = self.env.step(high_level_action_str, params)
                total_reward += reward
                steps += 1

            total_rewards.append(total_reward)
            steps_list.append(steps)
            print(f"[Eval] Episode {ep+1}: Reward = {total_reward:.2f}, Steps = {steps}")

        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards) # Standard deviation of rewards
        avg_steps = np.mean(steps_list)

        print(f"\n⮞ Durchschnittlicher Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"⮞ Durchschnittliche Schritte: {avg_steps:.2f}")
        
        if plot:
            plt.figure(figsize=(10, 5))
            plt.subplot(2, 1, 1)
            plt.plot(total_rewards, 'r-o', label='Episode Reward')
            plt.fill_between(
                range(len(total_rewards)),
                np.array(total_rewards) - np.std(std_reward),
                np.array(total_rewards) + np.std(std_reward),
                color='red', alpha=0.1
            )
            plt.title('Model Evaluation Results')
            plt.ylabel('Total Reward')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.subplot(2, 1, 2)
            plt.bar(range(len(steps_list)), steps_list, color='blue', alpha=0.6)
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.show()


    def evaluateProblem(self, problem, max_steps=2000, loop_detection=True, exploration_rate=0.1, save_to_file=False):
        """
        @brief Solves a specific problem with the trained model
        @param problem Path to the problem JSON file
        @param max_steps Maximum number of steps to avoid infinite loops (default: 2000)
        @param loop_detection Enables detection and avoidance of action loops (default: True)
        @param exploration_rate Probability of choosing a random action to break out of loops (default: 0.1)
        @param save_to_file Saves results to TXT file (default: False)
        @return tuple containing action sequence, parameters, and execution info
        """
        import time
        
        # Start time measurement
        start_time = time.time()
        
        obs = self.env.reset_specific_problem(problem)
        self.ppo_agent.load_models()

        isTerminal = False
        action_trace = []
        steps = 0
        
        # List to capture hash values of all visited states
        visited_states = []
        # Store hash value of environment state instead of observation
        visited_states.append(hash(str(self.env.state)))
        
        # For loop detection
        action_history = []
        repetition_count = {}
        last_action = None
        
        # Temperature for Boltzmann exploration (increases with repeated actions)
        temperature = 1.0

        print("Problem wird gelöst: " + problem)

        while not isTerminal and steps < max_steps:
            steps += 1
            # Get action probabilities from agent
            _, _, _, dist = self.ppo_agent.choose_action(obs)
            probs = dist.probs.detach().cpu().numpy()
            
            # Loop detection: Check if we're stuck in an action loop
            if loop_detection and len(action_history) >= 6:
                # Check last 6 actions for repeated patterns
                last_6_actions = ''.join([str(a) for a in action_history[-6:]])
                for pattern_length in [2, 3]:  # Search for 2- or 3-patterns
                    if len(last_6_actions) >= pattern_length*2:
                        pattern = last_6_actions[-pattern_length*2:-pattern_length]
                        if pattern == last_6_actions[-pattern_length:]:
                            #print(f"[LOOP DETECTED] Muster: {pattern}")
                            # Increase temperature to break out of loop
                            temperature = min(10.0, temperature * 1.5)  # Increase temperature, but not above 10
                            #print(f"Temperatur auf {temperature:.2f} erhöht")
                            
            # Decide whether to explore (random action) or exploit (best action)
            if np.random.random() < exploration_rate or temperature > 1.5:  # Increased exploration at high temperature
                # Exploration: Choose action based on Boltzmann distribution or randomly
                valid_actions = self.get_valid_actions(obs)
                if valid_actions:
                    if temperature > 1.2:
                        # Boltzmann exploration with current temperature
                        # Normalize probabilities and apply temperature
                        valid_probs = np.array([probs[a] for a in valid_actions])
                        if np.sum(valid_probs) > 0:
                            scaled_probs = np.exp(np.log(valid_probs + 1e-10) / temperature)
                            scaled_probs = scaled_probs / np.sum(scaled_probs)
                            high_level_action = np.random.choice(valid_actions, p=scaled_probs)
                            #print(f"[BOLTZMANN EXPLORATION] Temp={temperature:.2f}")
                        else:
                            high_level_action = np.random.choice(valid_actions)
                    else:
                        # Simple random exploration
                        high_level_action = np.random.choice(valid_actions)
                        
                    high_level_action_str = self.action_mapping[high_level_action]
                    #print(f"[EXPLORATION] Wähle: {high_level_action_str}")
                else:
                    # No valid actions available
                    print(f"Keine gültigen Aktionen für diesen Zustand möglich. Problem: {problem}")
                    return
            else:
                # Exploitation: Normal process with best action
                # Probabilities were already retrieved above
                
                # Check which actions are valid
                valid_actions = self.get_valid_actions(obs)
                
                # If no valid actions exist, end episode
                if not valid_actions:
                    print(f"Keine gültigen Aktionen für diesen Zustand möglich. Problem: {problem}")
                    return
    
                # Choose best valid action from available ones
                # Create mask for valid actions
                valid_mask = np.zeros_like(probs)
                for valid_action in valid_actions:
                    valid_mask[valid_action] = 1
                    
                # Multiply probabilities with mask and choose best
                masked_probs = probs * valid_mask
                if np.sum(masked_probs) > 0:
                    high_level_action = np.argmax(masked_probs)
                else:
                    # Fallback: Choose randomly from valid actions
                    high_level_action = np.random.choice(valid_actions)
                    
                high_level_action_str = self.action_mapping[high_level_action]

            # Try at most all actions
            tried_actions = set()
            while not self.env.check_action_execution(high_level_action_str, obs):
                tried_actions.add(high_level_action)
                
                if len(tried_actions) >= len(self.action_mapping):
                    print(f"Alle Aktionen probiert, keine ist gültig. Problem: {problem}")
                    return

                # Set probability of current action to 0
                probs[high_level_action] = 0.0
                
                # If all remaining probabilities are 0
                if np.all(probs == 0):
                    # Choose randomly from actions not yet tried
                    untried_actions = [i for i in range(len(self.action_mapping)) if i not in tried_actions]
                    if untried_actions:
                        high_level_action = np.random.choice(untried_actions)
                    else:
                        print("Keine gültige Aktion verfügbar. PROBLEM STUCK!")
                        return
                else:
                    # Choose action with highest probability
                    high_level_action = np.argmax(probs)
                
                high_level_action_str = self.action_mapping[high_level_action]

            # Heuristic parameter decision
            action_name, params = decide_parameters(obs, high_level_action_str)
            if action_name == "None":
                root = MCTSNode(state=self.env.state, action=(high_level_action_str, None))
                mcts = MCTS(root, depth=3, n_simulations=3)  # Reduced parameters for faster execution
                best_node = mcts.search()
                if best_node:
                    params = best_node.action[1]

            # Execute action
            obs, reward, isTerminal = self.env.step(high_level_action_str, params)
            
            # Add current state as hash to list
            visited_states.append(hash(str(self.env.state)))

            # Store action and parameters
            action_trace.append((high_level_action_str, params))
            
            # For loop detection: Store action in history
            action_history.append(high_level_action)
            
            # Detect special patterns (e.g., alternating stack/unstack)
            if last_action is not None:
                action_pair = (last_action, high_level_action)
                if action_pair in repetition_count:
                    repetition_count[action_pair] += 1
                    # If pattern is repeated too often, increase temperature
                    if repetition_count[action_pair] > 3:  # After 3 repetitions
                        temperature = min(5.0, temperature + 0.5)
                        #print(f"[PATTERN DETECTED] {self.action_mapping[action_pair[0]]} -> {self.action_mapping[action_pair[1]]}")
                        #print(f"Temperatur auf {temperature:.2f} erhöht")
                else:
                    repetition_count[action_pair] = 1
                    
            # Store current action for next iteration
            last_action = high_level_action
            
            # Cool down temperature over time if no patterns are detected
            if temperature > 1.0:
                temperature = max(1.0, temperature - 0.1)

        # Output results
        print("\n" + "="*50)
        print(f"ERGEBNIS FÜR PROBLEM: {problem}")
        print(f"Anzahl Schritte: {steps}/{max_steps}")
        print(f"Erfolgreicher Abschluss: {'Ja' if isTerminal else 'Nein - Maximale Schritte erreicht'}")
        print("="*50)
            
        # Statistics of actions
        action_counts = {}
        for action, _ in action_trace:
            if action in action_counts:
                action_counts[action] += 1
            else:
                action_counts[action] = 1
                
        print("\nAktionsstatistik:")
        for action, count in action_counts.items():
            print(f"{action}: {count} ({count/len(action_trace)*100:.1f}%)")
            
        initial_state_count = len(visited_states)
        # Loop-Detection and removal of unnecessary states
        if loop_detection:
            state_count = len(visited_states)

            for i in range(state_count):
                if i >= state_count:
                    break
                index = -1
                for j in range(1, state_count - i -1):
                    #print(j+i)
                    if j + i >= len(visited_states):
                        break
                    if visited_states[i] == visited_states[j + i]:
                        index = j + i
                if index != -1:
                    del visited_states[i : index]
                    del action_trace[i : index]
                    state_count -= (index - i - 1)
    
        print("\n" + "="*50)
        print("Anzahl der Aktionen nach Post-Processing:", len(action_trace), "\nOptimierung/Reduktion:" , f"{(1 - len(action_trace)/steps) * 100: .2f}", "%")
        print("="*50)        

        # End time measurement
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Format time readably
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.2f} Sekunden"
            elif seconds < 3600:
                minutes = int(seconds // 60)
                secs = seconds % 60
                return f"{minutes} Min {secs:.1f} Sek"
            else:
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = seconds % 60
                return f"{hours} Std {minutes} Min {secs:.1f} Sek"
        
        formatted_time = format_time(execution_time)
        print(f"\nBenötigte Zeit: {formatted_time}")

        # Calculate optimized action statistics
        optimized_action_counts = {}
        for action, _ in action_trace:
            if action in optimized_action_counts:
                optimized_action_counts[action] += 1
            else:
                optimized_action_counts[action] = 1
        
        print("\nOptimierte Aktionsstatistik:")
        for action, count in optimized_action_counts.items():
            percentage = count/len(action_trace)*100 if len(action_trace) > 0 else 0
            print(f"{action}: {count} ({percentage:.1f}%)")

        # Save results to file if desired
        if save_to_file:
            self._save_results_to_file(problem, steps, max_steps, isTerminal, action_trace, optimized_action_counts, len(action_trace), steps, execution_time, formatted_time)

        # Return extended with visited_states (all visited states)
        return isTerminal, len(action_trace), visited_states
    

    def _save_results_to_file(self, problem, steps, max_steps, is_terminal, action_trace, action_counts, optimized_steps, original_steps, execution_time, formatted_time):
        """
        @brief Saves the results of problem solving to a formatted TXT file
        @param problem Path to the problem JSON file
        @param steps Number of steps performed
        @param max_steps Maximum number of steps
        @param is_terminal Whether the problem was successfully solved
        @param action_trace List of performed actions with parameters
        @param action_counts Dictionary with action counts (after optimization)
        @param optimized_steps Number of steps after optimization
        @param original_steps Original number of steps
        @param execution_time Required time in seconds
        @param formatted_time Formatted time as string
        """
        import os
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        output_dir = "results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Extract problem name for filename
        problem_name = os.path.basename(problem).replace('.json', '')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/solution_{problem_name}_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            # Header
            f.write("="*70 + "\n")
            f.write("BELUGA CHALLENGE - LÖSUNGSPROTOKOLL\n")
            f.write("="*70 + "\n\n")
            
            # Problem information
            f.write(f"Problem: {problem}\n")
            f.write(f"Lösungsdatum: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
            f.write(f"Anzahl Schritte: {steps}/{max_steps}\n")
            f.write(f"Erfolgreicher Abschluss: {'Ja' if is_terminal else 'Nein - Maximale Schritte erreicht'}\n")
            f.write(f"Benötigte Zeit: {formatted_time}\n\n")
            
            # Action statistics (after optimization)
            f.write("="*70 + "\n")
            f.write("AKTIONSSTATISTIK (NACH OPTIMIERUNG)\n")
            f.write("="*70 + "\n\n")
            
            for action, count in action_counts.items():
                percentage = count/len(action_trace)*100 if len(action_trace) > 0 else 0
                f.write(f"{action:<25}: {count:>4} ({percentage:>5.1f}%)\n")
            
            # Optimization
            f.write(f"\n{'='*70}\n")
            f.write("OPTIMIERUNG\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"Ursprüngliche Anzahl Schritte: {original_steps}\n")
            f.write(f"Optimierte Anzahl Schritte: {optimized_steps}\n")
            optimization_percentage = (1 - optimized_steps/original_steps) * 100 if original_steps > 0 else 0
            f.write(f"Optimierung/Reduktion: {optimization_percentage:.2f}%\n\n")
            
            # Optimized action sequence
            f.write("="*70 + "\n")
            f.write("OPTIMIERTE AKTIONSSEQUENZ\n")
            f.write("="*70 + "\n\n")
            
            for i, (action, params) in enumerate(action_trace, 1):
                # Format parameters for better readability
                formatted_params = self._format_parameters(action, params)
                
                # Format output
                if formatted_params:
                    params_str = ", ".join([f"{k}={v}" for k, v in formatted_params.items()])
                    f.write(f"{i:>3}: {action:<25} | Parameter: {params_str}\n")
                else:
                    f.write(f"{i:>3}: {action:<25} | Parameter: -\n")
            
            f.write(f"\n{'='*70}\n")
            f.write("ENDE DES PROTOKOLLS\n")
            f.write(f"{'='*70}\n")
        
        print(f"\nErgebnisse wurden gespeichert in: {filename}")

    def _format_parameters(self, action, params):
        """
        @brief Formats parameters for better readability in output
        @param action The action name
        @param params The parameters to format
        @return Dictionary with formatted parameters
        
        Converts tuples and lists into meaningful dictionary formats.
        Filters out None values and 'none' keys.
        """
        # If params is already a dictionary, filter out None values and 'none' keys
        if isinstance(params, dict):
            # Filter out None values and 'none' keys
            filtered_params = {k: v for k, v in params.items() 
                             if v is not None and k.lower() != 'none'}
            return filtered_params
            
        # If params is a list or tuple, convert depending on action
        if isinstance(params, (list, tuple)):
            if len(params) == 0:
                return {}
            elif action in ["left_stack_rack", "right_stack_rack"]:
                if len(params) >= 2:
                    result = {"rack": params[0], "trailer": params[1]}
                else:
                    result = {"rack": params[0] if len(params) > 0 else None}
                # Filter out None values
                return {k: v for k, v in result.items() if v is not None}
            elif action in ["left_unstack_rack", "right_unstack_rack"]:
                if len(params) >= 2:
                    result = {"rack": params[0], "trailer": params[1]}
                else:
                    result = {"rack": params[0] if len(params) > 0 else None}
                # Filter out None values
                return {k: v for k, v in result.items() if v is not None}
            elif action == "load_beluga":
                if len(params) >= 1:
                    result = {"trailer": params[0]}
                else:
                    result = {}
                # Filter out None values
                return {k: v for k, v in result.items() if v is not None}
            elif action == "unload_beluga":
                return {}
            elif action in ["get_from_hangar", "deliver_to_hangar"]:
                if len(params) >= 2:
                    result = {"hangar": params[0], "trailer": params[1]}
                else:
                    result = {"hangar": params[0] if len(params) > 0 else None}
                # Filter out None values
                return {k: v for k, v in result.items() if v is not None}
            else:
                # Fallback for unknown actions
                return {"params": params}
        
        # Fallback for other types
        return params