from rl.env.environment import * # Environment 
from rl.agents.high_level.ppo_agent import * # High-Level-Agent
from rl.training.trainer import * # Trainer

if __name__ == '__main__':
    # Initialize environment
    env = Env(path="problemset2/")

    # Initialize High-Level-Agent (PPO)
    n_actions = 8  # Number of actions the agent can take
    batch_size = 30
    n_epochs = 10
    alpha = 0.0003
    ppo_agent = PPOAgent(n_actions=n_actions, batch_size=batch_size, alpha=alpha,
                         n_epochs=n_epochs, input_dims=30)

    # Initialize Trainer
    trainer = Trainer(env=env, ppo_agent=ppo_agent)

    # Start training
    trainer.train(n_episodes=20000, N=1000, max_steps_per_episode = 200)