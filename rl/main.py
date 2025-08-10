"""!
@file main.py
@brief Main entry point for the Beluga Challenge reinforcement learning solution

This module provides a command-line interface for training, evaluating, and testing
the agents on the Beluga Challenge shipping container optimization problem.
"""

from rl.env.environment import * # Environment 
from rl.agents.high_level.ppo_agent import * # High-Level-Agent
from rl.training.trainer import * # Trainer
import argparse

def main():
    """!
    @brief Main function for PPO training and evaluation of the Beluga Challenge
    
    Parses command line arguments and executes the appropriate mode:
    - train: Train the PPO agent
    - eval: Evaluate trained model performance
    - problem: Evaluate agent on specific problem instances
    """
    # Define command line arguments
    parser = argparse.ArgumentParser(description='PPO Training and Evaluation for Beluga Challenge')
    
    # Main mode
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'problem'], default='train',
                       help='Mode: train (Training), eval (Model Evaluation), problem (Problem Evaluation)')
    
    # Training parameters
    parser.add_argument('--train_old_models', action='store_true', default=True,
                       help='Load existing models (default: True)')
    parser.add_argument('--use_permutation', action='store_true', default=False,
                       help='Use observation permutation (default: False)')
    parser.add_argument('--n_episodes', type=int, default=10000,
                       help='Number of training episodes (default: 10000)')
    parser.add_argument('--base_index', type=int, default=61,
                       help='Base index for problem selection (default: 61)')
    
    # Evaluation parameters
    parser.add_argument('--n_eval_episodes', type=int, default=10,
                       help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--max_steps', type=int, default=200,
                       help='Maximum steps per episode (default: 200)')
    parser.add_argument('--plot', action='store_true', default=False,
                       help='Show plot after evaluation (default: False)')
    
    # Problem evaluation parameters
    parser.add_argument('--problem_path', type=str, default="problems/problem_90_s132_j137_r8_oc81_f43.json",
                       help='Path to problem for evaluation (default: problems/problem_90_s132_j137_r8_oc81_f43.json (Biggest Problem with 10 Racks))')
    parser.add_argument('--max_problem_steps', type=int, default=20000,
                       help='Maximum steps for problem evaluation (default: 20000)')
    parser.add_argument('--save_to_file', action='store_true', default=False,
                       help='Save results to TXT file (default: False)')
    
    args = parser.parse_args()
    
    # Initialize environment
    env = Env(path="problems/", base_index=args.base_index)

    # Initialize High-Level Agent (PPO)
    n_actions = 8  # Number of actions the agent can take
    batch_size = 128  # Increased batch size for more stable training
    n_epochs = 5     # Reduced number of epochs to prevent overfitting
    alpha = 0.0005   # Increased learning rate for faster learning
    N = 1024         # Buffer size
    ppo_agent = PPOAgent(n_actions=n_actions, batch_size=batch_size, alpha=alpha,
                         n_epochs=n_epochs, input_dims=40, policy_clip=0.2, N=N, model_name="ppo")

    # Initialize Trainer
    trainer = Trainer(env=env, ppo_agent=ppo_agent, debug=False)

    # Execute appropriate action based on mode
    if args.mode == 'train':
        print(f"Starting training with {args.n_episodes} episodes...")
        trainer.train(n_episodes=args.n_episodes, N=10, max_steps_per_episode=args.max_steps, 
                     train_on_old_models=args.train_old_models, use_permutation=args.use_permutation, 
                     start_learn_after=250)
        
    elif args.mode == 'eval':
        print(f"Starting model evaluation with {args.n_eval_episodes} episodes...")
        trainer.evaluateModel(n_eval_episodes=args.n_eval_episodes, 
                             max_steps_per_episode=args.max_steps, plot=args.plot)
        
    elif args.mode == 'problem':
        print(f"Evaluating problem: {args.problem_path}")
        trainer.evaluateProblem(args.problem_path, max_steps=args.max_problem_steps, save_to_file=args.save_to_file)

if __name__ == '__main__':
    main()

    
"""

Beispiele für Kommandozeilenaufrufe:

python -m rl.main --help

python -m rl.main --mode train

python -m rl.main --mode eval

python -m rl.main --mode problem

und die jeweiligen Argumente müssen angepasst werden.

Beispiele: 

python -m rl.main --mode train --n_episodes 5000 --base_index 61 --use_permutation 

python -m rl.main --mode problem --problem_path "problems/problem_90_s132_j137_r8_oc81_f43.json"

python -m rl.main --mode problem --problem_path "problems/problem_7_s49_j5_r2_oc85_f6.json" --save_to_file


"""