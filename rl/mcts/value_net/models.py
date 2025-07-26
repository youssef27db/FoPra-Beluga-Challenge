"""
models.py

Neural Network Models für MCTS Value Network

Enthält die Kern-Modellarchitekturen ohne Training-Logik oder Datenmanagement.
Fokus auf saubere, wiederverwendbare Netzwerk-Definitionen.

Klassen:
- SelfAttention: Multi-Head Self-Attention mit Padding-Maskierung
- MCTSValueNet: Value Network mit optionaler Attention-Schicht
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SelfAttention(nn.Module):
    """
    Multi-Head Self-Attention Modul mit Padding-Maskierung
    
    Ermöglicht dem Modell, Beziehungen zwischen verschiedenen Features/Positionen
    im Input-State zu lernen. Besonders nützlich für strukturierte Spiel-States
    wie Brettspiele, wo räumliche oder Feature-Abhängigkeiten wichtig sind.
    
    Args:
        embed_dim (int): Dimensionalität der Embeddings
        num_heads (int): Anzahl der Attention-Heads für Multi-Head-Attention
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super(SelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            padding_mask: Boolean mask [batch_size, seq_len] wo True = Padding Position
        
        Returns:
            output: Attention output [batch_size, seq_len, embed_dim]
        """
        # PyTorch MultiheadAttention erwartet [seq_len, batch_size, embed_dim]
        x = x.permute(1, 0, 2)
        
        if padding_mask is not None:
            output, _ = self.mha(x, x, x, key_padding_mask=padding_mask)
        else:
            output, _ = self.mha(x, x, x)
            
        # Zurück zu [batch_size, seq_len, embed_dim]
        return output.permute(1, 0, 2)


class MCTSValueNet(nn.Module):
    """
    Value Network für MCTS
    
    Schätzt den erwarteten Wert eines gegebenen Game-States. Das Netzwerk kann
    optional Self-Attention verwenden, um komplexe Feature-Abhängigkeiten zu
    modellieren. Padding-Maskierung ermöglicht variable Input-Längen.
    
    Architektur:
    1. Optional: Feature-wise Embedding + Self-Attention mit Padding-Mask
    2. Feed-Forward Layers mit Batch-Normalization und Dropout
    3. Regression-Output (ein Wert für State-Value)
    
    Args:
        input_dim (int): Anzahl der Input-Features
        hidden_dims (List[int]): Größen der Hidden Layers
        attention_dim (int): Embedding-Dimension für Attention (0 = keine Attention)
        num_heads (int): Anzahl Attention-Heads
        dropout_rate (float): Dropout-Rate für Regularisierung
        padding_value (float): Wert der für Padding verwendet wird
        use_batch_norm (bool): Ob Batch-Normalization verwendet werden soll
    """
    
    def __init__(self, 
                 input_dim: int = 112, 
                 hidden_dims: list = [256, 256, 128], 
                 attention_dim: int = 64, 
                 num_heads: int = 4, 
                 dropout_rate: float = 0.2, 
                 padding_value: float = -11.0, 
                 use_batch_norm: bool = True):
        super(MCTSValueNet, self).__init__()
        
        self.padding_value = padding_value
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.use_batch_norm = use_batch_norm
        
        # Attention-basierte Verarbeitung (optional)
        self.use_attention = attention_dim > 0
        if self.use_attention:
            # Jedes Input-Feature einzeln embedden für Attention
            self.embedding = nn.Linear(1, attention_dim)
            self.attention = SelfAttention(attention_dim, num_heads)
            # Attention-Output zu ersten Hidden Layer
            self.attn_output = nn.Linear(input_dim * attention_dim, hidden_dims[0])
        else:
            # Standard Input Layer ohne Attention
            self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        
        # Hidden Layers mit optionaler Batch-Normalization
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims[i + 1]))
        
        # Output Layer - ein Wert für State-Value
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
        # Regularisierung
        self.dropout = nn.Dropout(dropout_rate)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier/Glorot Gewichts-Initialisierung für bessere Konvergenz"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def _create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Erstellt Boolean-Maske für Padding-Positionen
        
        Args:
            x: Input tensor [batch_size, seq_len]
        
        Returns:
            padding_mask: Boolean mask [batch_size, seq_len] wo True = Padding
        """
        return x == self.padding_value
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass durch das Value Network
        
        Args:
            x: Input states [batch_size, input_dim]
            
        Returns:
            Predicted values [batch_size, 1]
        """
        batch_size = x.shape[0]
        
        if self.use_attention:
            # Attention-basierter Pfad
            padding_mask = self._create_padding_mask(x)
            
            # Features einzeln embedden: [batch, input_dim, 1] -> [batch, input_dim, attention_dim]
            x_reshaped = x.view(batch_size, self.input_dim, 1)
            x_embedded = self.embedding(x_reshaped)
            
            # Self-Attention mit Padding-Maskierung
            attn_output = self.attention(x_embedded, padding_mask)
            
            # Flatten und zu Hidden Layer projizieren
            attn_output = attn_output.reshape(batch_size, -1)
            x = F.relu(self.attn_output(attn_output))
        else:
            # Standard Feed-Forward Pfad
            x = F.relu(self.input_layer(x))
        
        x = self.dropout(x)
        
        # Hidden Layers mit Batch-Norm und optionalen Residual Connections
        for i, layer in enumerate(self.hidden_layers):
            identity = x if i > 0 and x.shape == layer(x).shape else None
            
            x = layer(x)
            if self.use_batch_norm and batch_size > 1:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            
            # Residual Connection wenn Dimensionen passen
            if identity is not None:
                x = x + identity
                
            x = self.dropout(x)
        
        # Output: Direkter Wert ohne Aktivierung (Regression)
        return self.output_layer(x)