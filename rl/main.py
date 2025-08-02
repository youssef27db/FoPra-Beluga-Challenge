from rl.env.environment import * # Environment 
from rl.agents.high_level.ppo_agent import * # High-Level-Agent
from rl.training.trainer import * # Trainer

if __name__ == '__main__':
    # Initialize environment
    env = Env(path="problems/", base_index=57)  # Set base_index to -1 for initial problem selection #TODO: Bei train old models = true, setzen wir base_index = 13

    # Initialize High-Level-Agent (PPO)
    n_actions = 8  # Number of actions the agent can take
    batch_size = 128  # Erhöhte Batch-Größe für stabileres Training
    n_epochs = 5     # Reduzierte Epochenanzahl gegen Overfitting
    alpha = 0.0005   # Erhöhte Lernrate für schnelleres Lernen
    N = 1024         # Buffer-Größe
    ppo_agent = PPOAgent(n_actions=n_actions, batch_size=batch_size, alpha=alpha,
                         n_epochs=n_epochs, input_dims=40, policy_clip=0.2, N=N)

    # Initialize Trainer
    trainer = Trainer(env=env, ppo_agent=ppo_agent, debug=False)

    # Start training
    trainer.train(n_episodes=10000, N=10, max_steps_per_episode=200, train_on_old_models=True, use_permutation=False, start_learn_after=250)

    # Evaluation
    #trainer.evaluateModel(n_eval_episodes=10, max_steps_per_episode=200, plot=True)

    # Problem solving
    # trainer.evaluateProblem("problemset2/problem_13.json")
    # trainer.evaluateProblem("problems/problem_90_s132_j137_r8_oc81_f43.json", max_steps=100000)