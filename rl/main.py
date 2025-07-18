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
                         n_epochs=n_epochs, input_dims=30, policy_clip=0.2)

    # Initialize Trainer
    trainer = Trainer(env=env, ppo_agent=ppo_agent)

    # Start training
    trainer.train(n_episodes=100000, N=20, max_steps_per_episode = 200, train_on_old_models=True)

    # Evaluation
    #trainer.evaluateModel(n_eval_episodes=10, max_steps_per_episode=200, plot=True)

    # Problem solving
    #trainer.evaluateProblem("problemset2/problem_12.json")