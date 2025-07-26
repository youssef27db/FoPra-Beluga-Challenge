"""
utils.py

Utility Functions für MCTS Value Network

Enthält Hilfsfunktionen für Datenverarbeitung und allgemeine Utilities.
Standalone Funktionen ohne Klassenabhängigkeiten.

Funktionen:
- preprocess_observations_with_padding: Variable-Length State Preprocessing
- create_device: Automatische Device-Auswahl
- set_random_seeds: Reproduzierbare Experimente
"""

import torch
import numpy as np
import random
from typing import Union, List, Optional


def preprocess_observations_with_padding(observations_batch: Union[List, torch.Tensor], 
                                       max_length: Optional[int] = None, 
                                       padding_value: float = -11.0) -> torch.Tensor:
    """
    Hilfsfunktion für Variable-Length State Preprocessing
    
    Konvertiert States unterschiedlicher Länge zu einheitlicher Größe durch Padding.
    Nützlich wenn verschiedene Game States unterschiedliche Anzahl Features haben.
    
    Args:
        observations_batch: Liste/Tensor von States mit variabler Länge
        max_length: Ziel-Länge nach Padding (None = auto-detect)
        padding_value: Wert für Padding-Positionen
        
    Returns:
        Tensor mit einheitlicher Länge [batch_size, max_length]
    """
    # Input zu Liste konvertieren für einheitliche Verarbeitung
    if isinstance(observations_batch, torch.Tensor):
        observations_list = [obs.cpu().numpy() for obs in observations_batch]
    else:
        observations_list = observations_batch
        
    if len(observations_list) == 0:
        return torch.empty(0, max_length or 0, dtype=torch.float32)
        
    # Maximale Länge bestimmen falls nicht gegeben
    if max_length is None:
        max_length = max(len(obs) for obs in observations_list)
    
    padded_obs_list = []
    for obs in observations_list:
        current_length = len(obs)
        
        if current_length >= max_length:
            # Kürzen wenn zu lang
            padded_obs = obs[:max_length]
        else:
            # Padding hinzufügen wenn zu kurz
            padding_needed = max_length - current_length
            if isinstance(obs, np.ndarray):
                padded_obs = np.concatenate([
                    obs, 
                    np.full(padding_needed, padding_value, dtype=obs.dtype)
                ])
            else:
                padded_obs = list(obs) + [padding_value] * padding_needed
        
        padded_obs_list.append(padded_obs)
    
    return torch.tensor(padded_obs_list, dtype=torch.float32)


def create_device(prefer_cuda: bool = True) -> str:
    """
    Automatische Device-Auswahl basierend auf Verfügbarkeit
    
    Args:
        prefer_cuda: Ob CUDA bevorzugt werden soll falls verfügbar
        
    Returns:
        Device string ('cuda' oder 'cpu')
    """
    if prefer_cuda and torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = 'cpu'
        if prefer_cuda:
            print("CUDA not available, using CPU")
        else:
            print("Using CPU device")
    
    return device


def set_random_seeds(seed: int = 42):
    """
    Setzt Random Seeds für reproduzierbare Experimente
    
    Args:
        seed: Random Seed Wert
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Für deterministische CUDA Operationen (kann Performance beeinträchtigen)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seeds set to {seed}")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Zählt Parameter in einem PyTorch Modell
    
    Args:
        model: PyTorch Modell
        trainable_only: Nur trainierbare Parameter zählen
        
    Returns:
        Anzahl Parameter
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def format_number(num: Union[int, float], precision: int = 2) -> str:
    """
    Formatiert große Zahlen für bessere Lesbarkeit
    
    Args:
        num: Zu formatierende Zahl
        precision: Nachkommastellen
        
    Returns:
        Formatierte Zahl als String
    """
    if num >= 1_000_000:
        return f"{num/1_000_000:.{precision}f}M"
    elif num >= 1_000:
        return f"{num/1_000:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def print_model_summary(model: torch.nn.Module, input_shape: tuple = None):
    """
    Druckt eine Übersicht über das Modell
    
    Args:
        model: PyTorch Modell
        input_shape: Eingabe-Shape für Modell (ohne batch dimension)
    """
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    print(f"Total parameters: {format_number(total_params)}")
    print(f"Trainable parameters: {format_number(trainable_params)}")
    print(f"Non-trainable parameters: {format_number(total_params - trainable_params)}")
    
    if input_shape:
        print(f"Input shape: {input_shape}")
        
        # Versuche Output-Shape zu bestimmen
        try:
            model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(1, *input_shape)
                output = model(dummy_input)
                print(f"Output shape: {tuple(output.shape[1:])}")
        except Exception as e:
            print(f"Could not determine output shape: {e}")
    
    print("=" * 60)
    print("ARCHITECTURE")
    print("=" * 60)
    print(model)
    print("=" * 60)


def validate_state_shape(state: Union[np.ndarray, torch.Tensor], expected_dim: int):
    """
    Validiert die Shape eines States
    
    Args:
        state: State Array/Tensor
        expected_dim: Erwartete Feature-Dimension
        
    Raises:
        ValueError: Wenn Shape nicht stimmt
    """
    if isinstance(state, torch.Tensor):
        state_shape = state.shape
    else:
        state_shape = np.array(state).shape
    
    if len(state_shape) == 1:
        actual_dim = state_shape[0]
    elif len(state_shape) == 2 and state_shape[0] == 1:
        actual_dim = state_shape[1]
    else:
        raise ValueError(f"Invalid state shape: {state_shape}. Expected 1D or (1, dim)")
    
    if actual_dim != expected_dim:
        raise ValueError(f"State dimension mismatch: got {actual_dim}, expected {expected_dim}")


def normalize_state(state: Union[np.ndarray, torch.Tensor], 
                   mean: Optional[Union[np.ndarray, torch.Tensor]] = None,
                   std: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalisiert einen State mit Z-Score Normalisierung
    
    Args:
        state: Zu normalisierender State
        mean: Mittelwerte für Normalisierung (None = aus State berechnen)
        std: Standardabweichungen für Normalisierung (None = aus State berechnen)
        
    Returns:
        Normalisierter State
    """
    if mean is None:
        mean = np.mean(state) if isinstance(state, np.ndarray) else torch.mean(state)
    
    if std is None:
        std = np.std(state) if isinstance(state, np.ndarray) else torch.std(state)
        
    # Vermeide Division durch Null
    if isinstance(std, (int, float)):
        std = max(std, 1e-8)
    else:
        std = np.maximum(std, 1e-8) if isinstance(std, np.ndarray) else torch.clamp(std, min=1e-8)
    
    return (state - mean) / std