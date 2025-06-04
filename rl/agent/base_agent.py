from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, state):
        """
        Wähle eine Aktion basierend auf dem gegebenen Zustand aus.
        """
        pass

    @abstractmethod
    def update(self, batch):
        """
        Update den Agenten mit einem Batch von Daten.
        """
        pass

    def save(self, filepath):
        """
        Optional: Speichere Modellparameter.
        """
        pass

    def load(self, filepath):
        """
        Optional: Lade Modellparameter.
        """
        pass
