"""
data_buffer.py

Datenmanagement für MCTS Value Network Training

Enthält Klassen für das Sammeln, Speichern und Verarbeiten von Trainingsdaten.
Fokus auf effiziente Datenverwaltung ohne ML-spezifische Logik.

Klassen:
- MCTSDataBuffer: Speicherung von State-Value-Paaren für Training
"""

import torch
import numpy as np
from collections import deque
import random
from typing import List, Tuple
import pickle


class MCTSDataBuffer:
    """
    Daten-Buffer für MCTS Value Network Training
    
    Speichert State-Value Paare die während MCTS-Simulationen gesammelt werden.
    Bietet Funktionen für Sampling, Normalisierung und Persistierung der Daten.
    
    Features:
    - Automatische Value-Normalisierung auf [-1, 1] Range
    - Efficient Sampling für Training
    - Speichern/Laden von Trainingsdaten
    - Memory-efficient Deque mit max. Größe
    
    Args:
        max_size (int): Maximale Anzahl gespeicherter Samples
        normalize_values (bool): Ob Values normalisiert werden sollen
    """
    
    def __init__(self, max_size: int = 100000, normalize_values: bool = True):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.normalize_values = normalize_values
        self.value_min = float('inf')
        self.value_max = float('-inf')
    
    def _normalize_value(self, value: float) -> float:
        """
        Normalisiert Values auf [-1, 1] Range für stabileres Training
        
        Args:
            value: Ursprünglicher Value
            
        Returns:
            Normalisierter Value im Bereich [-1, 1]
        """
        if not self.normalize_values:
            return value
            
        self.value_min = min(self.value_min, value)
        self.value_max = max(self.value_max, value)
        
        if self.value_max == self.value_min:
            return 0.0
            
        return 2.0 * ((value - self.value_min) / (self.value_max - self.value_min)) - 1.0
    
    def denormalize_value(self, normalized_value: float) -> float:
        """
        Konvertiert normalisierte Values zurück zur ursprünglichen Skala
        
        Args:
            normalized_value: Normalisierter Value
            
        Returns:
            Value in ursprünglicher Skala
        """
        if not self.normalize_values or self.value_max == self.value_min:
            return normalized_value
            
        return 0.5 * (normalized_value + 1.0) * (self.value_max - self.value_min) + self.value_min
    
    def add(self, observation, value: float):
        """
        Fügt ein State-Value Paar zum Buffer hinzu
        
        Args:
            observation: Game State (Tensor oder NumPy Array)
            value: Zugehöriger Value des States
        """
        # Tensor zu NumPy konvertieren für konsistente Speicherung
        if isinstance(observation, torch.Tensor):
            observation = observation.detach().cpu().numpy()
        elif isinstance(observation, np.ndarray):
            observation = observation.copy()
        
        normalized_value = self._normalize_value(value)
        self.buffer.append((observation, normalized_value))

    def add_batch(self, observations, values: List[float]):
        """
        Fügt mehrere State-Value Paare auf einmal hinzu
        
        Args:
            observations: Liste von Game States
            values: Liste von zugehörigen Values
        """
        for observation, value in zip(observations, values):
            self.add(observation, value)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sampelt einen Batch für Training
        
        Args:
            batch_size: Anzahl der zu samplenden Samples
        
        Returns:
            Tuple of:
            - observations_tensor: Batch von States [batch_size, input_dim]
            - values_tensor: Batch von Values [batch_size, 1]
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        observations, values = zip(*batch)
        
        observations_tensor = torch.FloatTensor(np.array(observations))
        values_tensor = torch.FloatTensor(values).unsqueeze(1)
        
        return observations_tensor, values_tensor
    
    def get_all_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gibt alle Daten im Buffer zurück
        
        Returns:
            Tuple of:
            - observations_tensor: Alle States [buffer_size, input_dim]
            - values_tensor: Alle Values [buffer_size, 1]
        """
        if len(self.buffer) == 0:
            return torch.empty(0, 112), torch.empty(0, 1)
        
        observations, values = zip(*self.buffer)
        observations_tensor = torch.FloatTensor(np.array(observations))
        values_tensor = torch.FloatTensor(values).unsqueeze(1)
        
        return observations_tensor, values_tensor
    
    def __len__(self) -> int:
        """Anzahl der Samples im Buffer"""
        return len(self.buffer)
    
    def clear(self):
        """Leert den Buffer komplett"""
        self.buffer.clear()
        # Reset normalization bounds
        self.value_min = float('inf')
        self.value_max = float('-inf')
    
    def get_stats(self) -> dict:
        """
        Gibt Statistiken über den Buffer zurück
        
        Returns:
            Dictionary mit Buffer-Statistiken
        """
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'value_min': None,
                'value_max': None,
                'value_range': None
            }
        
        return {
            'size': len(self.buffer),
            'value_min': self.value_min if self.value_min != float('inf') else None,
            'value_max': self.value_max if self.value_max != float('-inf') else None,
            'value_range': self.value_max - self.value_min if self.value_min != float('inf') else None,
            'normalize_values': self.normalize_values
        }
    
    def save(self, filepath: str):
        """
        Speichert Buffer in Datei
        
        Args:
            filepath: Pfad zur Zieldatei
        """
        save_data = {
            'buffer': list(self.buffer),
            'normalize_values': self.normalize_values,
            'value_min': self.value_min,
            'value_max': self.value_max,
            'max_size': self.max_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load(self, filepath: str):
        """
        Lädt Buffer aus Datei
        
        Args:
            filepath: Pfad zur Quelldatei
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        self.buffer = deque(data['buffer'], maxlen=self.max_size)
        self.normalize_values = data.get('normalize_values', True)
        self.value_min = data.get('value_min', float('inf'))
        self.value_max = data.get('value_max', float('-inf'))
        self.max_size = data.get('max_size', 100000)