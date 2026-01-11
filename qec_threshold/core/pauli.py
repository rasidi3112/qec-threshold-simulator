from enum import Enum
from dataclasses import dataclass


class PauliOperator(Enum):
    """Pauli operators representing fundamental quantum error types."""
    
    I = 0  # Identity (no error)
    X = 1  # Bit-flip error
    Y = 2  # Combined bit-flip and phase-flip
    Z = 3  # Phase-flip error

    def __mul__(self, other: 'PauliOperator') -> 'PauliOperator':
        """Pauli multiplication table (up to global phase)."""
        table = {
            (0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 3,
            (1, 0): 1, (1, 1): 0, (1, 2): 3, (1, 3): 2,
            (2, 0): 2, (2, 1): 3, (2, 2): 0, (2, 3): 1,
            (3, 0): 3, (3, 1): 2, (3, 2): 1, (3, 3): 0,
        }
        return PauliOperator(table[(self.value, other.value)])
    
    def __repr__(self) -> str:
        return f"PauliOperator.{self.name}"
    
    def __str__(self) -> str:
        return self.name


@dataclass
class QubitError:
    """Represents an error on a single qubit."""
    
    qubit_index: int
    error_type: PauliOperator
    time_step: int = 0
    
    def __repr__(self) -> str:
        return f"QubitError(qubit={self.qubit_index}, type={self.error_type.name}, t={self.time_step})"
