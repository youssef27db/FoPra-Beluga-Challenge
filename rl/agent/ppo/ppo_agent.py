from rl.agent.base_agent import BaseAgent
from ppo import Actor, Critic 


class PPOAgent(BaseAgent):
    def __init__(self, **kwargs):
        self.learning_rate = kwargs["lr"]
        self.clip_rate = kwargs["clip_rate"]
        self.actor = Actor(**kwargs)
        self.critic = Critic(**kwargs)


    def select_action(self, state):
        return self.actor.eval(state)



    def update(self, batch):
        pass