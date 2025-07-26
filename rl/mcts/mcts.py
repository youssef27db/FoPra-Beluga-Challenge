from .mcts_node import MCTSNode
from rl.env.state import *
from rl.env import *
from rl.utils.utils import debuglog
import random

class MCTS:
    def __init__(self, root: MCTSNode, depth: int = 5, n_simulations: int = 300, debug: bool = False):
        self.root = root
        self.depth = depth
        self.n_simulations = n_simulations
        self.debug = debug

    def search(self):
        terminal_node_found = False

        for i in range(self.n_simulations):
            if self.debug:
                print(f"\nIteration {i+1}/{self.n_simulations}")
            
            # 1. Selection
            node = self.select(self.root)
            if self.debug:
                print(f"Selected node: depth={node.depth}, action={node.action}")
            
            # 2. Expansion
            if not node.is_terminal():
                untried_actions = node.get_untried_action()
                if untried_actions:
                    action = random.choice(untried_actions)
                    if self.debug:
                        print(f"Expanding node with action: {action}")
                    node = node.expand(action)
                    
                    
                    if node.state.is_terminal():
                        if self.debug:
                            print("Terminal-Zustand erreicht! Lösung gefunden.")
                        terminal_node_found = True
                        # Reward wird durch evaluate() bereits gesetzt
                        reward = node.state.evaluate(node.depth)
                        node.backpropagate(reward)
                        if self.debug:
                            print(f"Rollout reward: {reward}")
                        break  # MCTS abbrechen
                else:
                    # Abbruch, wenn keine untried actions verfügbar sind
                    if not node.children or node.depth >= self.depth - 1:
                        if self.debug:
                            print(f"Keine weiteren Aktionen möglich bei Tiefe {node.depth}. Breche MCTS ab.")
                        # Final selection wie am Ende der Methode
                        debuglog("\nFinal selection (early):")
                        best_child = self.root.best_child(exploration_weight=0)
                        return best_child
                    else:
                        if self.debug:
                            print("No untried actions available, skipping expansion.")
            
            # 3. Simulation
            reward = self.rollout(node)
            if self.debug:
                print(f"Rollout reward: {reward}")
            
            # 4. Backpropagation
            node.backpropagate(reward)
    
        # Final selection - immer gleich, egal wie wir hierher gekommen sind
        debuglog("\nFinal selection:")
        best_child = self.root.best_child(exploration_weight=0)
        if best_child is None:
            debuglog("WARNING: Root has no children!")
            return None
        else:
            debuglog(f"Best child: action={best_child.action}, visits={best_child.visits}, reward={best_child.total_reward/best_child.visits if best_child.visits > 0 else 0}")
            if terminal_node_found:
                debuglog("Hinweis: Ein Terminal-Zustand wurde gefunden!")
            return best_child


    def select(self, node):
        """Traversiere den Baum, bis wir einen nicht vollständig expandierten Knoten oder ein Terminal-Node finden."""
        current_depth = 0
        while not node.is_terminal() and node.is_fully_expanded() and current_depth < self.depth:
            next_node = node.best_child()
            if next_node is None:
                break  # Falls keine Kinder vorhanden
            node = next_node
            current_depth += 1
        return node

    def rollout(self, node):
        """Simulate random actions from the node until we reach a terminal state or max depth."""
        state: ProblemState = node.state.copy()
        depth = node.depth  # Start from the node's current depth
        
        #print(f"DEBUG - Starting rollout from depth {depth}")
        rollout_actions = []
        
        while not state.is_terminal() and depth < self.depth:
            # Get possible actions as (action_name, params) tuples
            possible_actions = state.get_possible_actions()
            
            if not possible_actions:
                debuglog(f"DEBUG - No possible actions at depth {depth}")
                break
            
            # Choose a random action
            action_name, params = random.choice(possible_actions)
            
            # Apply the action
            #print(f"DEBUG - Rollout action: {action_name} with params {params}")
            rollout_actions.append((action_name, params))
            
            # Apply action to state
            state.apply_action(action_name, params)
            depth += 1
        
        if self.debug:
            print(f"DEBUG - Rollout completed with {len(rollout_actions)} actions")
            print(f"DEBUG - Final rollout actions: {rollout_actions[:5]}{'...' if len(rollout_actions) > 5 else ''}")
        
        # Calculate reward based on final state
        reward = state.evaluate(depth)
        if self.debug:
            print(f"DEBUG - Rollout ended at depth {depth}, final reward: {reward}")
            # alle subgoals und deren Erfüllung ausgeben
            # self.belugas_unloaded
            # self.belugas_finished
            # self.production_lines_finished
            print(f"DEBUG - Subgoals: {state.belugas_unloaded} unloaded, {state.belugas_finished} finished, {state.production_lines_finished} production lines finished")
        
        # Nach dem Rollout:
        if state.is_terminal() and self.debug:
            print("Terminal-Zustand im Rollout erreicht!")
        return reward

    def get_best_path(self):
        """
        Gibt den Pfad der bestbesuchten Kindknoten ab der Wurzel zurück.
        """
        path = []
        node = self.root
        while True:
            best_child = node.best_child(exploration_weight=0)
            if best_child is None:
                break
            path.append(best_child.action)
            node = best_child
        return path

    def collect_training_data(self, max_samples=1000, min_visits=5):
        """
        Sammelt Trainingsdaten aus dem MCTS-Baum für das RL-Modell.
        
        Args:
            max_samples (int): Maximale Anzahl zu sammelnder Trainingsbeispiele
            min_visits (int): Minimale Anzahl von Besuchen, die ein Knoten haben muss
                             um als Trainingsdatum berücksichtigt zu werden
        
        Returns:
            list: Liste von Tupeln (observation, value), wobei:
                - observation: Die Beobachtung des Zustands (Tensor oder Liste)
                - value: Der normalisierte Wert (total_reward / visits)
        """
        training_data = []
        # Verwende eine Queue für eine Breitensuche
        nodes_to_process = [self.root]
        
        # Statistiken für Diagnose
        total_nodes_checked = 0
        nodes_below_threshold = 0
        
        while nodes_to_process and len(training_data) < max_samples:
            current_node = nodes_to_process.pop(0)
            total_nodes_checked += 1
            
            # Nur Knoten mit genügend Besuchen berücksichtigen (für stabilere Schätzungen)
            if current_node.visits >= min_visits:
                # Beobachtung des aktuellen Zustands abrufen
                observation = current_node.state.get_observation_low_level()
                
                # Wert berechnen (Q-Wert des Knotens)
                value = current_node.total_reward / current_node.visits if current_node.visits > 0 else 0
                
                # Als Trainingspaar hinzufügen
                training_data.append((observation, value))
            else:
                nodes_below_threshold += 1
            
            # Kinder zur Verarbeitung hinzufügen
            if current_node.children:
                nodes_to_process.extend(current_node.children)
        
        # Diagnose-Ausgabe
        print(f"Gesammelte Trainingsdaten: {len(training_data)} Samples")
        print(f"Geprüfte Knoten insgesamt: {total_nodes_checked}")
        print(f"Knoten unter min_visits={min_visits}: {nodes_below_threshold}")
        print(f"Anzahl Root-Kinder: {len(self.root.children)}")
        
        # Falls zu wenige Knoten gesammelt wurden, versuche es mit geringerem min_visits
        if len(training_data) < 10 and min_visits > 1:
            print(f"Zu wenige Trainingsdaten! Versuche mit min_visits=1...")
            return self.collect_training_data(max_samples=max_samples, min_visits=1)
            
        return training_data

    def count_total_nodes(self):
        """Count total number of nodes in the MCTS tree."""
        def count_nodes(node):
            count = 1  # Count current node
            for child in node.children:
                count += count_nodes(child)
            return count
        
        return count_nodes(self.root)
    
    def get_tree_depth(self):
        """Get the maximum depth of the MCTS tree."""
        def get_max_depth(node):
            if not node.children:
                return node.depth
            return max(get_max_depth(child) for child in node.children)
        
        return get_max_depth(self.root)
    
    def search_single_iteration(self):
        """Perform a single MCTS iteration (selection, expansion, rollout, backpropagation)."""
        # 1. Selection
        node = self.select(self.root)
        if self.debug:
            print(f"Selected node: depth={node.depth}, action={node.action}")
        
        # 2. Expansion
        if not node.is_terminal():
            untried_actions = node.get_untried_action()
            if untried_actions:
                action = random.choice(untried_actions)
                if self.debug:
                    print(f"Expanding node with action: {action}")
                node = node.expand(action)
                
                if node.state.is_terminal():
                    if self.debug:
                        print("Terminal-Zustand erreicht! Lösung gefunden.")
                    # Reward wird durch evaluate() bereits gesetzt
                    reward = node.state.evaluate(node.depth)
                    node.backpropagate(reward)
                    if self.debug:
                        print(f"Rollout reward: {reward}")
                    return True  # Terminal node found
        
        # 3. Rollout
        if not node.is_terminal():
            reward = self.rollout(node)
            if self.debug:
                print(f"Rollout reward: {reward}")
            
            # 4. Backpropagation
            node.backpropagate(reward)
        
        return False  # No terminal node found