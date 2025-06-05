# ziel sollte es sein, training und evaluation folgendermaßen zu starten:
# python main.py --mode train --model PPO --episodes 1000 --learning-rate 0.0003 
# python main.py --mode evaluate --model PPO --load-checkpoint ./checkpoints/ppo.pth

# eventuell noch weitere Argumente für: "display information", etc.

import argparse
import logging
from trainer import training, evaluation


def main():
    parser = argparse.ArgumentParser(description="Reinforcement Learning Training and Evaluation")
    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True, help='Mode to run: train or evaluate')
    parser.add_argument('--model', type=str, default='ppo', required=True, help='Model type (e.g., PPO, DQN)')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes for training or evaluation')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate for training')
    parser.add_argument('--load-checkpoint', type=str, help='Path to the checkpoint file for evaluation')
    parser.add_argument('--clip-rate', type=float, default=0.2, help='Clip rate for PPO')
    args = parser.parse_args()

    # update-intervall (Episoden) und batch_size (Steps) sollten als optionale Argumente hinzugefügt werden
    logging.basicConfig(level=logging.INFO)


    if args.mode == 'train':
        training.run(args)
    elif args.mode == 'evaluate':
        evaluation.run(args)
  