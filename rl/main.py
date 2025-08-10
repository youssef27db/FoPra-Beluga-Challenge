from rl.env.environment import * # Environment 
from rl.agents.high_level.ppo_agent import * # High-Level-Agent
from rl.training.trainer import * # Trainer
import argparse

def main():
    # Kommandozeilenargumente definieren
    parser = argparse.ArgumentParser(description='PPO Training und Evaluation für Beluga Challenge')
    
    # Hauptmodus
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'problem'], default='train',
                       help='Modus: train (Training), eval (Modell-Evaluation), problem (Problem-Evaluation)')
    
    # Training-Parameter
    parser.add_argument('--train_old_models', action='store_true', default=True,
                       help='Lade vorhandene Modelle (default: True)')
    parser.add_argument('--use_permutation', action='store_true', default=False,
                       help='Verwende Observation-Permutation (default: False)')
    parser.add_argument('--n_episodes', type=int, default=10000,
                       help='Anzahl Trainingsepisoden (default: 10000)')
    parser.add_argument('--base_index', type=int, default=61,
                       help='Base Index für Problem-Auswahl (default: 61)')
    
    # Evaluation-Parameter
    parser.add_argument('--n_eval_episodes', type=int, default=10,
                       help='Anzahl Evaluationsepisoden (default: 10)')
    parser.add_argument('--max_steps', type=int, default=200,
                       help='Maximale Schritte pro Episode (default: 200)')
    parser.add_argument('--plot', action='store_true', default=False,
                       help='Zeige Plot nach Evaluation (default: False)')
    
    # Problem-Evaluation
    parser.add_argument('--problem_path', type=str, default="problems/problem_90_s132_j137_r8_oc81_f43.json",
                       help='Pfad zum zu evaluierenden Problem (default: problems/problem_90_s132_j137_r8_oc81_f43.json (Bigest Problem with 10 Racks))')
    parser.add_argument('--max_problem_steps', type=int, default=20000,
                       help='Maximale Schritte für Problem-Evaluation (default: 20000)')
    parser.add_argument('--save_to_file', action='store_true', default=False,
                       help='Speichere Ergebnisse in TXT-Datei (default: False)')
    
    args = parser.parse_args()
    
    # Initialize environment
    env = Env(path="problems/", base_index=args.base_index)

    # Initialize High-Level-Agent (PPO)
    n_actions = 8  # Number of actions the agent can take
    batch_size = 128  # Erhöhte Batch-Größe für stabileres Training
    n_epochs = 5     # Reduzierte Epochenanzahl gegen Overfitting
    alpha = 0.0005   # Erhöhte Lernrate für schnelleres Lernen
    N = 1024         # Buffer-Größe
    ppo_agent = PPOAgent(n_actions=n_actions, batch_size=batch_size, alpha=alpha,
                         n_epochs=n_epochs, input_dims=40, policy_clip=0.2, N=N, model_name="ppo")

    # Initialize Trainer
    trainer = Trainer(env=env, ppo_agent=ppo_agent, debug=False)

    # Führe je nach Modus die entsprechende Aktion aus
    if args.mode == 'train':
        print(f"Starte Training mit {args.n_episodes} Episoden...")
        trainer.train(n_episodes=args.n_episodes, N=10, max_steps_per_episode=args.max_steps, 
                     train_on_old_models=args.train_old_models, use_permutation=args.use_permutation, 
                     start_learn_after=250)
        
    elif args.mode == 'eval':
        print(f"Starte Modell-Evaluation mit {args.n_eval_episodes} Episoden...")
        trainer.evaluateModel(n_eval_episodes=args.n_eval_episodes, 
                             max_steps_per_episode=args.max_steps, plot=args.plot)
        
    elif args.mode == 'problem':
        print(f"Evaluiere Problem: {args.problem_path}")
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