from env.environment import * # Environment 
from agents.high_level.ppo_agent import * # High-Level-Agent
from training.trainer import * # Trainer

if __name__ == '__main__':
    # Initialize environment
    env = Env(path="problemset2/")

    # Initialize High-Level-Agent (PPO)
    n_actions = 8  # Number of actions the agent can take
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    ppo_agent = PPOAgent(n_actions=n_actions, batch_size=batch_size, alpha=alpha,
                         n_epochs=n_epochs, input_dims=30)

    # Initialize Trainer
    trainer = Trainer(env=env, ppo_agent=ppo_agent)

    # Start training
    trainer.train(n_episodes=1000, N=20)