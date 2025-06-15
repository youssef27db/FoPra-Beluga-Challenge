from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    def __init__(self, env, agent, **kwargs):
        self.env = env
        self.agent = agent
        self.episodes = kwargs['episodes'] # arg parse kann hier genutzt werden, um die Anzahl der Episoden zu setzen
    
    @abstractmethod
    def train(self):
        pass

    def evaluate(self, episodes=10):
        """
        Führt eine Evaluation durch (z.B. mehrere Episoden ohne Lernen).
        """
        total_reward = 0
        for _ in range(self.episodes):
            done = False
            state = self.env.state
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done = self.env.step(action)
                state = next_state
                total_reward += reward
            state = self.env.reset() # hier müsste neues problem geladen/gestartet werden
        avg_reward = total_reward / episodes
        print(f"Average evaluation reward: {avg_reward}")
        return avg_reward