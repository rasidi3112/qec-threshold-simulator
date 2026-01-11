from typing import List
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pauli import PauliOperator


class DepolarizingChannel:
    """Depolarizing noise channel with symmetric X/Y/Z error rates."""
    
    def __init__(self, error_probability: float):
        if not 0 <= error_probability <= 1:
            raise ValueError(f"Error probability must be in [0, 1], got {error_probability}")
        
        self.p = error_probability
        self.px = error_probability / 3
        self.py = error_probability / 3
        self.pz = error_probability / 3
    
    def apply(self, num_qubits: int, rng: np.random.Generator) -> List[PauliOperator]:
        """Apply depolarizing noise to all qubits."""
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
        return f"DepolarizingChannel(p={self.p:.4f})"
