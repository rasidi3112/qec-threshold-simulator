from typing import List
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pauli import PauliOperator
from core.surface_code import SurfaceCode


class CorrelatedNoiseChannel:
    """Spatially correlated noise model."""
    
    def __init__(self, error_probability: float, correlation_length: float = 1.0):
        if not 0 <= error_probability <= 1:
            raise ValueError(f"Error probability must be in [0, 1], got {error_probability}")
        if correlation_length <= 0:
            raise ValueError(f"Correlation length must be positive, got {correlation_length}")
        
        self.p = error_probability
        self.xi = correlation_length
    
    def apply(self, code: SurfaceCode, rng: np.random.Generator) -> List[PauliOperator]:
        """Apply correlated noise to the surface code."""
        d = code.distance
        errors = [PauliOperator.I] * code.num_data_qubits
        
        num_events = rng.poisson(self.p * code.num_data_qubits)
        error_types = [PauliOperator.X, PauliOperator.Y, PauliOperator.Z]
        
        for _ in range(num_events):
            primary_row = rng.integers(0, d)
            primary_col = rng.integers(0, d)
            primary_error = error_types[rng.integers(0, 3)]
            
            for row in range(d):
                for col in range(d):
                    dist = abs(row - primary_row) + abs(col - primary_col)
                    prob = np.exp(-dist / self.xi)
                    
                    if rng.random() < prob:
                        idx = code.data_qubit_positions[(row, col)]
                        errors[idx] = errors[idx] * primary_error
        
        return errors
    
    def __repr__(self) -> str:
        return f"CorrelatedNoiseChannel(p={self.p:.4f}, xi={self.xi:.2f})"
