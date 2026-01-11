from typing import List
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pauli import PauliOperator


class BiasedNoiseChannel:
    """Biased noise channel with configurable X/Z asymmetry."""
    
    def __init__(self, error_probability: float, bias_ratio: float = 1.0):
        if not 0 <= error_probability <= 1:
            raise ValueError(f"Error probability must be in [0, 1], got {error_probability}")
        if bias_ratio <= 0:
            raise ValueError(f"Bias ratio must be positive, got {bias_ratio}")
        
        self.p = error_probability
        self.eta = bias_ratio
        
        self.px = self.p / (2 + self.eta)
        self.py = self.px
        self.pz = self.eta * self.px
    
    def apply(self, num_qubits: int, rng: np.random.Generator) -> List[PauliOperator]:
        """Apply biased noise to all qubits."""
        errors = []
        
        for _ in range(num_qubits):
            r = rng.random()
            
            if r < self.px:
                errors.append(PauliOperator.X)
            elif r < self.px + self.py:
                errors.append(PauliOperator.Y)
            elif r < self.p:
                errors.append(PauliOperator.Z)
            else:
                errors.append(PauliOperator.I)
        
        return errors
    
    def __repr__(self) -> str:
        return f"BiasedNoiseChannel(p={self.p:.4f}, eta={self.eta:.1f})"
