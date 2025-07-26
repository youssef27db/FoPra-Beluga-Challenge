"""
trainer.py

Training Pipeline für MCTS Value Network

Enthält die komplette Training-Logik mit modernen ML-Praktiken.
Fokus auf robuste, wiederverwendbare Training-Pipeline.

Klassen:
- MCTSValueNetTrainer: Training und Evaluation des Value Networks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple
from .data_buffer import MCTSDataBuffer


class MCTSValueNetTrainer:
    """
    Trainer für MCTS Value Network
    
    Bietet komplette Training-Pipeline mit modernen ML-Praktiken:
    - Adam Optimizer mit Learning Rate Scheduling
    - Early Stopping basierend auf Validation Loss
    - Gradient Clipping gegen exploding gradients
    - Checkpoint-System für Training-Wiederaufnahme
    - Validation Split mit konsistenter Aufteilung
    
    Args:
        value_net: Das zu trainierende MCTSValueNet Modell
        learning_rate: Lernrate für Adam Optimizer
        batch_size: Batch-Größe für Training
        device: 'cpu' oder 'cuda'
        weight_decay: L2-Regularisierung
    """
    
    def __init__(self, 
                 value_net: nn.Module,
                 learning_rate: float = 0.001,
                 batch_size: int = 64,
                 device: str = 'cpu',
                 weight_decay: float = 1e-4):
        
        self.value_net = value_net.to(device)
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Optimizer mit Weight Decay für Regularisierung
        self.optimizer = optim.Adam(
            self.value_net.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Learning Rate Scheduler für adaptive Lernrate
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=50, verbose=True,
            threshold=0.0001, min_lr=1e-6
        )
        
        self.criterion = nn.MSELoss()
        
        # Training-Statistiken
        self.training_losses = []
        self.validation_losses = []
        self.training_step = 0
        
        # Data splits (werden während Training gesetzt)
        self.train_states = None
        self.train_values = None
        self.val_states = None
        self.val_values = None
    
    def train_step(self, states: torch.Tensor, values: torch.Tensor) -> float:
        """
        Führt einen Trainings-Schritt durch
        
        Args:
            states: Batch von States [batch_size, input_dim]
            values: Batch von Values [batch_size, 1]
            
        Returns:
            Loss-Wert für diesen Schritt
        """
        states = states.to(self.device)
        values = values.to(self.device)
        
        self.optimizer.zero_grad()
        
        predictions = self.value_net(states)
        loss = self.criterion(predictions, values)
        
        loss.backward()
        
        # Gradient Clipping gegen instabile Gradienten
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.training_step += 1
        
        return loss.item()
    
    def validate(self, states: torch.Tensor, values: torch.Tensor) -> float:
        """
        Evaluiert das Modell auf Validation Set
        
        Args:
            states: Validation States [val_size, input_dim]
            values: Validation Values [val_size, 1]
            
        Returns:
            Validation Loss
        """
        self.value_net.eval()
        
        with torch.no_grad():
            states = states.to(self.device)
            values = values.to(self.device)
            
            predictions = self.value_net(states)
            loss = self.criterion(predictions, values)
            
            return loss.item()
    
    def _prepare_data_splits(self, data_buffer: MCTSDataBuffer, validation_split: float) -> bool:
        """
        Bereitet einmalige Datenaufteilung vor
        
        Args:
            data_buffer: Buffer mit Trainingsdaten
            validation_split: Anteil für Validation
            
        Returns:
            True wenn erfolgreich, False wenn keine Daten
        """
        all_states, all_values = data_buffer.get_all_data()
        
        if len(all_states) == 0:
            return False
        
        # Shuffle und Split
        indices = torch.randperm(len(all_states))
        all_states = all_states[indices]
        all_values = all_values[indices]
        
        val_size = max(1, int(len(all_states) * validation_split))
        self.train_states = all_states[val_size:]
        self.train_values = all_values[val_size:]
        self.val_states = all_states[:val_size]
        self.val_values = all_values[:val_size]
        
        return True
    
    def train(self, 
              data_buffer: MCTSDataBuffer, 
              epochs: int = 100,
              validation_split: float = 0.1,
              early_stopping_patience: int = 100,
              min_delta: float = 1e-4,
              verbose: bool = True,
              save_interval: int = 10,
              checkpoint_prefix: str = "checkpoint") -> dict:
        """
        Haupttraining-Schleife mit Early Stopping
        
        Args:
            data_buffer: Buffer mit Trainingsdaten
            epochs: Maximale Anzahl Epochen
            validation_split: Anteil der Daten für Validation
            early_stopping_patience: Epochen ohne Verbesserung bis Stop
            min_delta: Minimale Verbesserung für Early Stopping
            verbose: Ob Training-Progress geloggt werden soll
            save_interval: Alle X Epochen Checkpoint speichern
            checkpoint_prefix: Prefix für Checkpoint-Dateien
        
        Returns:
            training_history: Dictionary mit Training-Statistiken
        """
        
        # Datenaufteilung vorbereiten
        if not self._prepare_data_splits(data_buffer, validation_split):
            print("No data in buffer!")
            return {'train_losses': [], 'val_losses': [], 'epochs_trained': 0}
        
        if verbose:
            print(f"Training set size: {len(self.train_states)}")
            print(f"Validation set size: {len(self.val_states)}")
        
        # Training-Variablen initialisieren
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'epochs_trained': 0,
            'learning_rates': []
        }
        
        # Haupttraining-Schleife
        for epoch in range(epochs):
            # Training Phase
            self.value_net.train()
            train_losses = []
            
            # Shuffle Training Data für jede Epoche
            train_indices = torch.randperm(len(self.train_states))
            shuffled_train_states = self.train_states[train_indices]
            shuffled_train_values = self.train_values[train_indices]
            
            # Batch-weise Training
            for i in range(0, len(shuffled_train_states), self.batch_size):
                batch_states = shuffled_train_states[i:i + self.batch_size]
                batch_values = shuffled_train_values[i:i + self.batch_size]
                
                if len(batch_states) == 0:  # Skip empty batches
                    continue
                    
                loss = self.train_step(batch_states, batch_values)
                train_losses.append(loss)
            
            if not train_losses:  # No training happened
                print("No training batches processed!")
                break
                
            avg_train_loss = np.mean(train_losses)
            
            # Validation Phase
            val_loss = self.validate(self.val_states, self.val_values)
            self.scheduler.step(val_loss)
            
            # Statistiken speichern
            training_history['train_losses'].append(avg_train_loss)
            training_history['val_losses'].append(val_loss)
            training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Progress Logging
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train={avg_train_loss:.6f}, Val={val_loss:.6f}, LR={self.optimizer.param_groups[0]['lr']:.8f}")
            
            # Best Model Tracking
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().detach().clone() for k, v in self.value_net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Checkpoint Speichern
            if (epoch + 1) % save_interval == 0:
                checkpoint_path = f"{checkpoint_prefix}_epoch_{epoch+1}.pth"
                self.save_checkpoint(checkpoint_path, epoch + 1, training_history)
                if verbose and epoch > 0:  # Don't spam on first checkpoint
                    print(f"Checkpoint saved: {checkpoint_path}")
            
            # Early Stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        # Finales Checkpoint
        final_checkpoint_path = f"{checkpoint_prefix}_final_epoch_{epoch+1}.pth"
        self.save_checkpoint(final_checkpoint_path, epoch + 1, training_history)
        
        if verbose:
            print(f"Training completed: {epoch+1} epochs")
            print(f"Best validation loss: {best_val_loss:.6f}")
            print(f"Final checkpoint saved: {final_checkpoint_path}")
        
        # Bestes Modell wiederherstellen
        if best_model_state is not None:
            self.value_net.load_state_dict(best_model_state)
            if verbose:
                print("Restored best model weights")
            
        training_history['epochs_trained'] = epoch + 1
        return training_history
    
    def predict(self, state: np.ndarray, data_buffer: Optional[MCTSDataBuffer] = None) -> float:
        """
        Vorhersage für einen einzelnen State
        
        Args:
            state: Game State als NumPy Array
            data_buffer: Optional für Value-Denormalisierung
        
        Returns:
            Predicted Value für den State
        """
        self.value_net.eval()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            prediction = self.value_net(state_tensor)
            pred_value = prediction.item()
            
            # Denormalisierung falls Buffer verfügbar
            if data_buffer is not None and data_buffer.normalize_values:
                pred_value = data_buffer.denormalize_value(pred_value)
                
            return pred_value
    
    def evaluate(self, data_buffer: MCTSDataBuffer) -> dict:
        """
        Evaluiert das Modell auf allen Daten im Buffer
        
        Args:
            data_buffer: Buffer mit Test-Daten
            
        Returns:
            Dictionary mit Evaluation-Metriken
        """
        all_states, all_values = data_buffer.get_all_data()
        
        if len(all_states) == 0:
            return {'mse': float('inf'), 'mae': float('inf'), 'samples': 0}
        
        self.value_net.eval()
        
        with torch.no_grad():
            all_states = all_states.to(self.device)
            all_values = all_values.to(self.device)
            
            predictions = self.value_net(all_states)
            
            mse = nn.MSELoss()(predictions, all_values).item()
            mae = nn.L1Loss()(predictions, all_values).item()
            
            return {
                'mse': mse,
                'mae': mae,
                'samples': len(all_states),
                'rmse': np.sqrt(mse)
            }
    
    def save_checkpoint(self, filepath: str, epoch: int, training_history: dict):
        """
        Speichert umfassendes Training-Checkpoint
        
        Args:
            filepath: Pfad zur Checkpoint-Datei
            epoch: Aktuelle Epoche
            training_history: Training-Verlauf
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.value_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': training_history,
            'training_step': self.training_step,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str) -> Tuple[int, dict]:
        """
        Lädt Checkpoint und ermöglicht Training-Fortsetzung
        
        Args:
            filepath: Pfad zur Checkpoint-Datei
            
        Returns:
            Tuple of (epoch, training_history)
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.value_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_step = checkpoint['training_step']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch'], checkpoint['training_history']
    
    def save_model(self, filepath: str):
        """
        Speichert nur das trainierte Modell (ohne Training-State)
        
        Args:
            filepath: Pfad zur Modell-Datei
        """
        torch.save({
            'model_state_dict': self.value_net.state_dict(),
            'model_config': {
                'input_dim': getattr(self.value_net, 'input_dim', None),
                'attention_dim': getattr(self.value_net, 'attention_dim', None),
                'padding_value': getattr(self.value_net, 'padding_value', None)
            }
        }, filepath)
    
    def load_model(self, filepath: str):
        """
        Lädt nur das Modell (ohne Training-State)
        
        Args:
            filepath: Pfad zur Modell-Datei
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.value_net.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully")
    
    def get_training_stats(self) -> dict:
        """
        Gibt aktuelle Training-Statistiken zurück
        
        Returns:
            Dictionary mit Training-Statistiken
        """
        return {
            'training_steps': self.training_step,
            'current_lr': self.optimizer.param_groups[0]['lr'],
            'train_losses': self.training_losses.copy(),
            'val_losses': self.validation_losses.copy(),
            'best_val_loss': min(self.validation_losses) if self.validation_losses else float('inf')
        }