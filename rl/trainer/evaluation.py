# trainer/evaluation.py
from agent.ppo.ppo_agent import PPOAgent
from env.env import MyEnv

AGENTS = {
    "ppo": PPOAgent,
    # "dqn": DQNAgent, ...
}

def run(args):
    # kann mit trainer.evaluate(args) umgesetzt werden, nur wird zusätzlich ein bereits trainiertes modell hier für den agenten geladen
    pass
