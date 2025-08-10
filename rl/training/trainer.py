import numpy as np
from rl.env.environment import * # Environment 
from rl.agents.high_level.ppo_agent import *  # High-Level-Agent
from rl.agents.low_level.heuristics import *  # Low-Level-Heuristik
from rl.mcts import *  # MCTS-Algorithmus
from rl.utils.utils import *
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, env: Env, ppo_agent: PPOAgent, mcts_params=None, debug=False):
        self.env = env
        self.ppo_agent: PPOAgent = ppo_agent  # High-Level-Agent
        self.mcts = None
        self.debug = debug  # Debug-Modus für zusätzliche Ausgaben
        
        # Tracking-Metriken
        self.episode_rewards = []
        self.avg_rewards = []
        self.steps_per_episode = []
        self.best_score = -90000
        self.score_history = []
        self.learn_iters = 0
        self.invalid_action_counts = {i: 0 for i in range(8)}  # Zähler für ungültige Aktionen nach Typ
        
        # Exploration-Parameter
        self.epsilon_start = 0.9  # Anfangswert für Epsilon (Explorations-Wahrscheinlichkeit)
        self.epsilon_end = 0.2   # Endwert für Epsilon
        self.epsilon_decay = 0.00001  # Rate, mit der Epsilon reduziert wird
        self.total_steps = 0      # Gesamtzahl der durchgeführten Schritte
        
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
        """Überprüft, welche Aktionen im aktuellen Zustand gültig sind"""
        valid_actions = []
        for action_idx in range(len(self.action_mapping)):
            if self.env.check_action_execution(self.action_mapping[action_idx], obs):
                valid_actions.append(action_idx)
        return valid_actions

    def train(self, n_episodes=2000, N=5, max_steps_per_episode = 200, train_on_old_models = False, start_learn_after = 500, use_permutation = False):
        """
        Trainiert den Agenten über eine bestimmte Anzahl von Episoden.
        
        Args:
            n_episodes: Anzahl der Trainingsepisoden
            N: Frequenz der Lernschritte
            max_steps_per_episode: Maximale Anzahl von Schritten pro Episode
            train_on_old_models: Ob vorhandene Modelle geladen werden sollen
            start_learn_after: Nach wie vielen Schritten das Lernen beginnen soll
            use_permutation: Ob Beobachtungen permutiert werden sollen (kann Training stabilisieren, kostet aber Zeit)
        """
        if train_on_old_models:
            self.ppo_agent.load_models()  # Lade die Modelle des PPO-Agenten
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
                # High-Level-Entscheidung (PPO)
                # Berechne aktuelle Epsilon-Wert für Exploration
                epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                          np.exp(-self.epsilon_decay * self.total_steps)
                
                # Epsilon-Greedy Strategie für Exploration
                if np.random.random() < epsilon:
                    # Explorative Aktion: Wähle eine zufällige gültige Aktion
                    valid_actions = self.get_valid_actions(obs)
                    if valid_actions:
                        high_level_action = np.random.choice(valid_actions)
                        high_level_action_str = self.action_mapping[high_level_action]
                        
                        # Um die PPO-Logik beizubehalten, brauchen wir die Verteilung
                        if use_permutation:
                            _, prob, val, dist = self.ppo_agent.choose_action(permute_high_level_observation(np.random.permutation(10), obs))
                        else:
                            _, prob, val, dist = self.ppo_agent.choose_action(obs)
                    else:
                        # Falls keine gültigen Aktionen vorhanden sind, normale Strategien nutzen
                        if use_permutation:
                            obs_ = None  # Reset der Beobachtung für die nächste Iteration
                            permutation = np.random.permutation(10)
                            permuted_obs = permute_high_level_observation(permutation, obs)
                            high_level_action, prob, val, dist = self.ppo_agent.choose_action(permuted_obs)
                        else:
                            # Ohne Permutation direkt die Beobachtung verwenden (schneller)
                            high_level_action, prob, val, dist = self.ppo_agent.choose_action(obs)
                        high_level_action_str = self.action_mapping[high_level_action]
                else:
                    # Exploitative Aktion: Nutze die PPO-Policy
                    if use_permutation:
                        obs_ = None  # Reset der Beobachtung für die nächste Iteration
                        permutation = np.random.permutation(10)
                        permuted_obs = permute_high_level_observation(permutation, obs)
                        high_level_action, prob, val, dist = self.ppo_agent.choose_action(permuted_obs)
                    else:
                        # Ohne Permutation direkt die Beobachtung verwenden (schneller)
                        high_level_action, prob, val, dist = self.ppo_agent.choose_action(obs)
                    high_level_action_str = self.action_mapping[high_level_action]  # Mapping der Aktion

                # Debug-Ausgaben deaktiviert für schnellere Ausführung
                # debuglog("-" *20)
                # debuglog(high_level_action_str)
                # debuglog("-" *20)

                probs = dist.probs.detach().cpu().numpy()
                
                # Prüfen, welche Aktionen gültig sind
                valid_actions = self.get_valid_actions(obs)
                
                # Falls keine gültigen Aktionen existieren, Fehlermeldung und Episode beenden
                if not valid_actions:
                    print(f"Keine gültigen Aktionen für diesen Zustand möglich. Problem: {self.env.problem_name}")
                    isTerminal = True
                    reward -= 5000.0  # Reduzierte Bestrafung, da es wirklich unmöglich ist
                    break
                
                # Aktion ausbalancieren - wenn zu viele unstack-Aktionen verwendet wurden,
                # reduziere deren Wahrscheinlichkeit zugunsten anderer Aktionen
                if episode > 100 and self.total_steps > 500:  # Nach einer Anlaufphase
                    total_invalid = sum(self.invalid_action_counts.values())
                    if total_invalid > 0:
                        unstack_percentage = (self.invalid_action_counts[6] + self.invalid_action_counts[7]) / total_invalid
                        if unstack_percentage > 0.4:  # Wenn mehr als 40% der ungültigen Aktionen unstacks sind
                            # Reduziere die Wahrscheinlichkeit für unstack-Aktionen
                            unstack_idx = [6, 7]  # left_unstack_rack, right_unstack_rack
                            scale_factor = 0.5  # Skaliere die Wahrscheinlichkeit nach unten
                            for idx in unstack_idx:
                                if idx < len(probs):
                                    probs[idx] *= scale_factor
                            # Normalisiere die Wahrscheinlichkeiten wieder
                            if np.sum(probs) > 0:
                                probs = probs / np.sum(probs)
                                
                            # Aktualisiere die Action-Wahl
                            high_level_action = np.argmax(probs)
                            high_level_action_str = self.action_mapping[high_level_action]
                            prob = probs[high_level_action]
                    
                # Debug-Ausgabe für gültige Aktionen, wenn aktiviert
                if self.debug and steps % 10 == 0:  # Nicht zu oft ausgeben
                    valid_action_names = [self.action_mapping[idx] for idx in valid_actions]
                    print(f"Gültige Aktionen: {valid_action_names}")

                # Versuche maximal alle Aktionen durchzuprobieren
                tried_actions = set()
                while not self.env.check_action_execution(high_level_action_str, obs):
                    # Zähle ungültige Aktionen für spätere Analyse
                    self.invalid_action_counts[high_level_action] += 1
                    tried_actions.add(high_level_action)
                    
                    if len(tried_actions) >= len(self.action_mapping):
                        print(f"Alle Aktionen probiert, keine ist gültig. Problem: {self.env.problem_name}")
                        isTerminal = True
                        reward -= 10000.0
                        break

                    # Setze die Wahrscheinlichkeit der aktuellen Aktion auf 0
                    probs[high_level_action] = 0.0
                    
                    # Wenn alle restlichen Wahrscheinlichkeiten 0 sind
                    if np.all(probs == 0):
                        # Wähle zufällig aus den noch nicht probierten Aktionen
                        untried_actions = [i for i in range(len(self.action_mapping)) if i not in tried_actions]
                        if untried_actions:
                            high_level_action = np.random.choice(untried_actions)
                        else:
                            print("No valid action found, stopping episode.")
                            isTerminal = True
                            reward -= 10000.0
                            break
                    else:
                        # Wähle die Aktion mit der höchsten Wahrscheinlichkeit
                        high_level_action = np.argmax(probs)
                    
                    high_level_action_str = self.action_mapping[high_level_action]  # Aktualisiere den Aktionsnamen
                    prob = probs[high_level_action]  # Aktualisiere die Wahrscheinlichkeit der Aktion

                if not isTerminal:
                    # Low-Level-Agent: 
                    # Heuristik
                    action_name, params = decide_parameters(obs, high_level_action_str)
                    # Wenn keine Heuristik gefunden wurde, dann MCTS verwenden (optimiert)
                    if action_name == "None":
                        root = MCTSNode(state=self.env.state, action=(high_level_action_str, None))

                        # MCTS mit diesem Root-Node starten - reduzierte Tiefe und Simulationen für Geschwindigkeit
                        mcts = MCTS(root, depth=5, n_simulations=60) #TODO 60 für große probleme
                        best_node = mcts.search()
                        
                        if best_node:
                            params = best_node.action[1]
                    else:
                        bool_heuristic = True

                    # Debug-Ausgaben deaktiviert für schnellere Ausführung
                    # debuglog("-" *20)
                    # debuglog(params)
                    # debuglog("-" *20)
            

                    # Überprüfe, ob wir eine loop haben mit unstacking und stacking
                    params_check = list(params.values()) if isinstance(params, dict) else list(params)
                    if params_check != []:
                        if (high_level_action == 4 and last_action == 6) or (high_level_action == 5 and last_action == 7) \
                            or (high_level_action == 6 and last_action == 4) or (high_level_action == 7 and last_action == 5):
                            if last_trailer_id == params_check[1] and last_rack_id == params_check[0]:
                                reward -= 200.0  # Reduzierte Bestrafung für Schleifen
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
                                # Bestrafe das Unstacking ohne Folgeaktion
                                reward -= 20.0
                        else:
                            # Andere Heuristiken erhalten kleinere Belohnungen
                            reward += 5.0
                    

                print_action = high_level_action_str  # Speichere die letzte Aktion für Debugging
                # Speichere Erfahrung für PPO
                self.ppo_agent.remember(obs, high_level_action, prob, val, reward, isTerminal)

                if reward > 0: 
                    positive_actions_reward += reward
                if not obs_ is None:
                    obs = obs_
                total_reward += reward
                steps += 1
                self.total_steps += 1

                # PPO-Lernschritt am Ende der Episode (optimierte Frequenz)
                if self.total_steps >= start_learn_after and self.total_steps % (N*2) == 0:
                    self.ppo_agent.learn()
                    self.learn_iters += 1

                # debuglog(steps) # Debug-Ausgabe deaktiviert
                if steps >= self.env.get_max_steps() or total_reward <= -10000:
                    isTerminal = True  # Angepasste Abbruchbedingung mit weniger strengem Reward-Limit
    
            # Metriken speichern
            self.episode_rewards.append(total_reward)
            avg_reward = np.mean(self.episode_rewards[-10:])
            self.avg_rewards.append(avg_reward)
            self.steps_per_episode.append(steps)
            
            # Überprüfen, ob ein Reset von Epsilon nötig ist
            # Wenn die letzten 6 Episoden alle sehr schlechte Rewards haben, setze Epsilon zurück
            if len(self.episode_rewards) >= 6:
                recent_rewards = self.episode_rewards[-6:]
                if all(reward <= -10000 for reward in recent_rewards):
                    print("\nSehr schlechte Performance in den letzten 10 Episoden. Setze Epsilon zurück, um mehr zu explorieren.")
                    self.epsilon_start = 0.9  # Zurücksetzen auf Anfangswert
                    self.epsilon_decay = 0.00001  # Zurücksetzen der Decay-Rate
                    self.total_steps = 0  # Zurücksetzen der Schritte für die Epsilon-Berechnung

            # Fortschritt anzeigen
            if avg_reward > self.best_score:
                self.ppo_agent.save_models()
                self.best_score = avg_reward

            # Prüfen, ob das Problem gelöst wurde
            solved = self.env.state.is_terminal()
            status_symbol = "✅" if solved else "  "
            
            print(f'{status_symbol} episode {episode}, score {total_reward:.1f}, avg score {avg_reward:.1f}, Best avg score {self.best_score:.1f}',
                  f'time_steps {steps}/{self.env.get_max_steps()}, learn_iters {self.learn_iters}, positive reward {positive_actions_reward:.1f}, problem {self.env.problem_name}, {self.env.base_index}')
                  
            # Speichere das Modell alle 100 Episoden als Checkpoint
            if episode > 0 and episode % 100 == 0:
                self.ppo_agent.save_models()
            
            # Zeige alle 20 Episoden Statistik zu ungültigen Aktionen an
            # if episode % 20 == 0 and episode > 0:
            #     total_invalid = sum(self.invalid_action_counts.values())
            #     if total_invalid > 0:
            #         print("\nStatistik ungültiger Aktionen:")
            #         for action_idx, count in sorted(self.invalid_action_counts.items(), key=lambda x: x[1], reverse=True):
            #             if count > 0:
            #                 percentage = (count / total_invalid) * 100
            #                 print(f"  {self.action_mapping[action_idx]}: {count} ({percentage:.1f}%)")
            #         print("")  # Leerzeile
                    
            #         # Zurücksetzen der Zähler alle 100 Episoden
            #         if episode % 100 == 0:
            #             self.invalid_action_counts = {i: 0 for i in range(8)}


    def evaluateModel(self, n_eval_episodes=10, max_steps_per_episode=200, plot = False):
        """
        Evaluiert das Modell über eine bestimmte Anzahl von Episoden.
        Gibt den durchschnittlichen Reward und die Standardabweichung aus.
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

                # Wähle Aktion ohne Lernen
                _, _, _, dist = self.ppo_agent.choose_action(obs)

                probs = dist.probs.detach().cpu().numpy() 
                high_level_action = np.argmax(probs)
                high_level_action_str = self.action_mapping[high_level_action]

                # Prüfe, welche Aktionen gültig sind
                valid_actions = self.get_valid_actions(obs)
                
                # Falls keine gültigen Aktionen existieren, Episode beenden
                if not valid_actions:
                    print(f"[Eval] Keine gültigen Aktionen für diesen Zustand. Problem: {self.env.problem_name}")
                    isTerminal = True
                    break

                # Versuche maximal alle Aktionen durchzuprobieren
                tried_actions = set()
                while not self.env.check_action_execution(high_level_action_str, obs):
                    tried_actions.add(high_level_action)
                    
                    if len(tried_actions) >= len(self.action_mapping):
                        print(f"[Eval] Alle Aktionen probiert, keine ist gültig. Problem: {self.env.problem_name}")
                        isTerminal = True
                        break

                    # Setze die Wahrscheinlichkeit der aktuellen Aktion auf 0
                    probs[high_level_action] = 0.0
                    
                    # Wenn alle restlichen Wahrscheinlichkeiten 0 sind
                    if np.all(probs == 0):
                        # Wähle zufällig aus den noch nicht probierten Aktionen
                        untried_actions = [i for i in range(len(self.action_mapping)) if i not in tried_actions]
                        if untried_actions:
                            high_level_action = np.random.choice(untried_actions)
                        else:
                            isTerminal = True
                            break
                    else:
                        # Wähle die Aktion mit der höchsten Wahrscheinlichkeit
                        high_level_action = np.argmax(probs)
                    
                    high_level_action_str = self.action_mapping[high_level_action]

                if isTerminal:
                    break

                # Low-Level-Agent
                action_name, params = decide_parameters(obs, high_level_action_str)
                if action_name == "None":
                    root = MCTSNode(state=self.env.state, action=(high_level_action_str, None))
                    mcts = MCTS(root, depth=3, n_simulations=3)  # Reduzierte Parameter für schnellere Ausführung
                    best_node = mcts.search()
                    if best_node:
                        params = best_node.action[1]

                # Loop-Prävention
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
        std_reward = np.std(total_rewards) # Standardabweichung der Belohnungen
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
            plt.title('Evaluierungsergebnisse vom Modell')
            plt.ylabel('Total Reward')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.subplot(2, 1, 2)
            plt.bar(range(len(steps_list)), steps_list, color='blue', alpha=0.6)
            plt.xlabel('Episode')
            plt.ylabel('Schritte')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.show()


    def evaluateProblem(self, problem, max_steps=2000, loop_detection=True, exploration_rate=0.1, save_to_file=False):
        """
        Löst ein spezifisches Problem mit dem trainierten Modell.
        
        Args:
            problem: Pfad zum Problem-JSON
            max_steps: Maximale Anzahl an Schritten, um Endlosschleifen zu vermeiden
            loop_detection: Aktiviert die Erkennung und Vermeidung von Aktionsschleifen
            exploration_rate: Wahrscheinlichkeit, eine zufällige Aktion zu wählen, um aus Schleifen auszubrechen
            save_to_file: Speichert Ergebnisse in TXT-Datei
            
        Ausgegeben wird die Reihenfolge der Aktionen und Parameter.
        """
        import time
        
        # Zeitmessung starten
        start_time = time.time()
        
        obs = self.env.reset_specific_problem(problem)
        self.ppo_agent.load_models()

        isTerminal = False
        action_trace = []
        steps = 0
        
        # Liste zur Erfassung der Hash-Werte aller besuchten Zustände
        visited_states = []
        # Speichere den Hash-Wert des Environment-Zustands statt der Beobachtung
        visited_states.append(hash(str(self.env.state)))
        
        # Für Loop-Detection
        action_history = []
        repetition_count = {}
        last_action = None
        
        # Temperatur für Boltzmann-Exploration (steigt bei wiederholten Aktionen)
        temperature = 1.0

        print("Problem wird gelöst: " + problem)

        while not isTerminal and steps < max_steps:
            steps += 1
            # Hole Action-Wahrscheinlichkeiten vom Agenten
            _, _, _, dist = self.ppo_agent.choose_action(obs)
            probs = dist.probs.detach().cpu().numpy()
            
            # Loop-Detection: Überprüfe, ob wir in einer Aktionsschleife stecken
            if loop_detection and len(action_history) >= 6:
                # Überprüfe die letzten 6 Aktionen auf wiederholte Muster
                last_6_actions = ''.join([str(a) for a in action_history[-6:]])
                for pattern_length in [2, 3]:  # Suche nach 2er- oder 3er-Mustern
                    if len(last_6_actions) >= pattern_length*2:
                        pattern = last_6_actions[-pattern_length*2:-pattern_length]
                        if pattern == last_6_actions[-pattern_length:]:
                            #print(f"[LOOP DETECTED] Muster: {pattern}")
                            # Erhöhe die Temperatur, um aus der Schleife auszubrechen
                            temperature = min(10.0, temperature * 1.5)  # Erhöhe die Temperatur, aber nicht über 10
                            #print(f"Temperatur auf {temperature:.2f} erhöht")
                            
            # Entscheide, ob exploriert werden soll (zufällige Aktion) oder ausgebeutet (beste Aktion)
            if np.random.random() < exploration_rate or temperature > 1.5:  # Erhöhte Exploration bei hoher Temperatur
                # Exploration: Wähle Aktion basierend auf Boltzmann-Verteilung oder zufällig
                valid_actions = self.get_valid_actions(obs)
                if valid_actions:
                    if temperature > 1.2:
                        # Boltzmann-Exploration mit aktueller Temperatur
                        # Normalisiere Wahrscheinlichkeiten und wende Temperatur an
                        valid_probs = np.array([probs[a] for a in valid_actions])
                        if np.sum(valid_probs) > 0:
                            scaled_probs = np.exp(np.log(valid_probs + 1e-10) / temperature)
                            scaled_probs = scaled_probs / np.sum(scaled_probs)
                            high_level_action = np.random.choice(valid_actions, p=scaled_probs)
                            #print(f"[BOLTZMANN EXPLORATION] Temp={temperature:.2f}")
                        else:
                            high_level_action = np.random.choice(valid_actions)
                    else:
                        # Einfache zufällige Exploration
                        high_level_action = np.random.choice(valid_actions)
                        
                    high_level_action_str = self.action_mapping[high_level_action]
                    #print(f"[EXPLORATION] Wähle: {high_level_action_str}")
                else:
                    # Keine gültigen Aktionen vorhanden
                    print(f"Keine gültigen Aktionen für diesen Zustand möglich. Problem: {problem}")
                    return
            else:
                # Exploitation: Normaler Prozess mit der besten Aktion
                # Wahrscheinlichkeiten wurden bereits oben geholt
                
                # Prüfe, welche Aktionen gültig sind
                valid_actions = self.get_valid_actions(obs)
                
                # Falls keine gültigen Aktionen existieren, Episode beenden
                if not valid_actions:
                    print(f"Keine gültigen Aktionen für diesen Zustand möglich. Problem: {problem}")
                    return
    
                # Wähle beste gültige Aktion aus den verfügbaren
                # Erstelle eine Maske für gültige Aktionen
                valid_mask = np.zeros_like(probs)
                for valid_action in valid_actions:
                    valid_mask[valid_action] = 1
                    
                # Multipliziere Wahrscheinlichkeiten mit der Maske und wähle die beste
                masked_probs = probs * valid_mask
                if np.sum(masked_probs) > 0:
                    high_level_action = np.argmax(masked_probs)
                else:
                    # Fallback: Wähle zufällig aus den gültigen Aktionen
                    high_level_action = np.random.choice(valid_actions)
                    
                high_level_action_str = self.action_mapping[high_level_action]

            # Versuche maximal alle Aktionen durchzuprobieren
            tried_actions = set()
            while not self.env.check_action_execution(high_level_action_str, obs):
                tried_actions.add(high_level_action)
                
                if len(tried_actions) >= len(self.action_mapping):
                    print(f"Alle Aktionen probiert, keine ist gültig. Problem: {problem}")
                    return

                # Setze die Wahrscheinlichkeit der aktuellen Aktion auf 0
                probs[high_level_action] = 0.0
                
                # Wenn alle restlichen Wahrscheinlichkeiten 0 sind
                if np.all(probs == 0):
                    # Wähle zufällig aus den noch nicht probierten Aktionen
                    untried_actions = [i for i in range(len(self.action_mapping)) if i not in tried_actions]
                    if untried_actions:
                        high_level_action = np.random.choice(untried_actions)
                    else:
                        print("Keine gültige Aktion verfügbar. PROBLEM STUCK!")
                        return
                else:
                    # Wähle die Aktion mit der höchsten Wahrscheinlichkeit
                    high_level_action = np.argmax(probs)
                
                high_level_action_str = self.action_mapping[high_level_action]

            # Heuristische Parameterentscheidung
            action_name, params = decide_parameters(obs, high_level_action_str)
            if action_name == "None":
                root = MCTSNode(state=self.env.state, action=(high_level_action_str, None))
                mcts = MCTS(root, depth=3, n_simulations=3)  # Reduzierte Parameter für schnellere Ausführung
                best_node = mcts.search()
                if best_node:
                    params = best_node.action[1]

            # Aktion ausführen
            obs, reward, isTerminal = self.env.step(high_level_action_str, params)
            
            # Aktuellen Zustand als Hash zur Liste hinzufügen
            visited_states.append(hash(str(self.env.state)))

            # Aktion und Parameter speichern
            action_trace.append((high_level_action_str, params))
            
            # Für Loop-Detection: Aktion in Verlauf speichern
            action_history.append(high_level_action)
            
            # Spezielle Muster erkennen (z.B. abwechselndes stack/unstack)
            if last_action is not None:
                action_pair = (last_action, high_level_action)
                if action_pair in repetition_count:
                    repetition_count[action_pair] += 1
                    # Wenn ein Muster zu oft wiederholt wird, erhöhe die Temperatur
                    if repetition_count[action_pair] > 3:  # Nach 3 Wiederholungen
                        temperature = min(5.0, temperature + 0.5)
                        #print(f"[PATTERN DETECTED] {self.action_mapping[action_pair[0]]} -> {self.action_mapping[action_pair[1]]}")
                        #print(f"Temperatur auf {temperature:.2f} erhöht")
                else:
                    repetition_count[action_pair] = 1
                    
            # Speichern der aktuellen Aktion für den nächsten Durchlauf
            last_action = high_level_action
            
            # Abkühlen der Temperatur über Zeit, wenn keine Muster erkannt werden
            if temperature > 1.0:
                temperature = max(1.0, temperature - 0.1)

            #print(f"Aktion: {high_level_action_str}, Parameter: {params}")

        
        # print("\nReihenfolge der Aktionen:")
        # for i, (action, params) in enumerate(action_trace, 1):
        #     # Formatiere Parameter für bessere Lesbarkeit
        #     formatted_params = self._format_parameters(action, params)
        #     print(f"{i:02d}: {action}  |  Parameter: {formatted_params}")

        # Ausgabe
        print("\n" + "="*50)
        print(f"ERGEBNIS FÜR PROBLEM: {problem}")
        print(f"Anzahl Schritte: {steps}/{max_steps}")
        print(f"Erfolgreicher Abschluss: {'Ja' if isTerminal else 'Nein - Maximale Schritte erreicht'}")
        print("="*50)
            
        # Statistische Auswertung
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

        # Zeitmessung beenden
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Formatiere die Zeit leserlich
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

        # Aktionsstatistik nach Optimierung berechnen
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

        # Speichere Ergebnisse in Datei, wenn gewünscht
        if save_to_file:
            self._save_results_to_file(problem, steps, max_steps, isTerminal, action_trace, optimized_action_counts, len(action_trace), steps, execution_time, formatted_time)

        # Rückgabe erweitert um visited_states (alle besuchten Zustände)
        return isTerminal, len(action_trace), visited_states
    

    def _save_results_to_file(self, problem, steps, max_steps, is_terminal, action_trace, action_counts, optimized_steps, original_steps, execution_time, formatted_time):
        """
        Speichert die Ergebnisse der Problemlösung in eine formatierte TXT-Datei.
        
        Args:
            problem: Pfad zum Problem-JSON
            steps: Anzahl der durchgeführten Schritte
            max_steps: Maximale Anzahl an Schritten
            is_terminal: Ob das Problem erfolgreich gelöst wurde
            action_trace: Liste der durchgeführten Aktionen mit Parametern
            action_counts: Dictionary mit Aktionszählungen (nach Optimierung)
            optimized_steps: Anzahl der Schritte nach Optimierung
            original_steps: Ursprüngliche Anzahl der Schritte
            execution_time: Benötigte Zeit in Sekunden
            formatted_time: Formatierte Zeit als String
        """
        import os
        from datetime import datetime
        
        # Erstelle Output-Verzeichnis falls es nicht existiert
        output_dir = "results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Extrahiere Problem-Namen für Dateinamen
        problem_name = os.path.basename(problem).replace('.json', '')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/solution_{problem_name}_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            # Header
            f.write("="*70 + "\n")
            f.write("BELUGA CHALLENGE - LÖSUNGSPROTOKOLL\n")
            f.write("="*70 + "\n\n")
            
            # Problem-Information
            f.write(f"Problem: {problem}\n")
            f.write(f"Lösungsdatum: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
            f.write(f"Anzahl Schritte: {steps}/{max_steps}\n")
            f.write(f"Erfolgreicher Abschluss: {'Ja' if is_terminal else 'Nein - Maximale Schritte erreicht'}\n")
            f.write(f"Benötigte Zeit: {formatted_time}\n\n")
            
            # Aktionsstatistik (nach Optimierung)
            f.write("="*70 + "\n")
            f.write("AKTIONSSTATISTIK (NACH OPTIMIERUNG)\n")
            f.write("="*70 + "\n\n")
            
            for action, count in action_counts.items():
                percentage = count/len(action_trace)*100 if len(action_trace) > 0 else 0
                f.write(f"{action:<25}: {count:>4} ({percentage:>5.1f}%)\n")
            
            # Optimierung
            f.write(f"\n{'='*70}\n")
            f.write("OPTIMIERUNG\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"Ursprüngliche Anzahl Schritte: {original_steps}\n")
            f.write(f"Optimierte Anzahl Schritte: {optimized_steps}\n")
            optimization_percentage = (1 - optimized_steps/original_steps) * 100 if original_steps > 0 else 0
            f.write(f"Optimierung/Reduktion: {optimization_percentage:.2f}%\n\n")
            
            # Optimierte Aktionssequenz
            f.write("="*70 + "\n")
            f.write("OPTIMIERTE AKTIONSSEQUENZ\n")
            f.write("="*70 + "\n\n")
            
            for i, (action, params) in enumerate(action_trace, 1):
                # Formatiere Parameter für bessere Lesbarkeit
                formatted_params = self._format_parameters(action, params)
                
                # Formatiere die Ausgabe
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
        Formatiert Parameter für bessere Lesbarkeit in der Ausgabe.
        Konvertiert Tupel und Listen in aussagekräftige Dictionary-Formate.
        """
        # Wenn params bereits ein Dictionary ist, filtere None-Werte und 'none'-Schlüssel heraus
        if isinstance(params, dict):
            # Filtere None-Werte und 'none'-Schlüssel heraus
            filtered_params = {k: v for k, v in params.items() 
                             if v is not None and k.lower() != 'none'}
            return filtered_params
            
        # Wenn params eine Liste oder Tupel ist, konvertiere je nach Aktion
        if isinstance(params, (list, tuple)):
            if len(params) == 0:
                return {}
            elif action in ["left_stack_rack", "right_stack_rack"]:
                if len(params) >= 2:
                    result = {"rack": params[0], "trailer": params[1]}
                else:
                    result = {"rack": params[0] if len(params) > 0 else None}
                # Filtere None-Werte heraus
                return {k: v for k, v in result.items() if v is not None}
            elif action in ["left_unstack_rack", "right_unstack_rack"]:
                if len(params) >= 2:
                    result = {"rack": params[0], "trailer": params[1]}
                else:
                    result = {"rack": params[0] if len(params) > 0 else None}
                # Filtere None-Werte heraus
                return {k: v for k, v in result.items() if v is not None}
            elif action == "load_beluga":
                if len(params) >= 1:
                    result = {"trailer": params[0]}
                else:
                    result = {}
                # Filtere None-Werte heraus
                return {k: v for k, v in result.items() if v is not None}
            elif action == "unload_beluga":
                return {}
            elif action in ["get_from_hangar", "deliver_to_hangar"]:
                if len(params) >= 2:
                    result = {"hangar": params[0], "trailer": params[1]}
                else:
                    result = {"hangar": params[0] if len(params) > 0 else None}
                # Filtere None-Werte heraus
                return {k: v for k, v in result.items() if v is not None}
            else:
                # Fallback für unbekannte Aktionen
                return {"params": params}
        
        # Fallback für andere Typen
        return params