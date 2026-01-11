from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

from .pauli import PauliOperator


@dataclass
class SurfaceCode:
    """Rotated Surface Code Implementation."""
    
    distance: int
    
    num_data_qubits: int = field(init=False)
    num_x_stabilizers: int = field(init=False)
    num_z_stabilizers: int = field(init=False)
    
    data_qubit_positions: Dict[Tuple[int, int], int] = field(default_factory=dict)
    x_stabilizer_positions: Dict[Tuple[int, int], int] = field(default_factory=dict)
    z_stabilizer_positions: Dict[Tuple[int, int], int] = field(default_factory=dict)
    
    x_stabilizer_qubits: Dict[int, List[int]] = field(default_factory=dict)
    z_stabilizer_qubits: Dict[int, List[int]] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.distance < 3 or self.distance % 2 == 0:
            raise ValueError(f"Distance must be an odd integer >= 3, got {self.distance}")
        self._initialize_lattice()
    
    def _initialize_lattice(self):
        d = self.distance
        
        self.num_data_qubits = d * d
        qubit_idx = 0
        for row in range(d):
            for col in range(d):
                self.data_qubit_positions[(row, col)] = qubit_idx
                qubit_idx += 1
        
        x_stab_idx = 0
        for row in range(d - 1):
            for col in range(d - 1):
                if (row + col) % 2 == 0:
                    self.x_stabilizer_positions[(row, col)] = x_stab_idx
                    qubits = []
                    for dr, dc in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                        pos = (row + dr, col + dc)
                        if pos in self.data_qubit_positions:
                            qubits.append(self.data_qubit_positions[pos])
                    self.x_stabilizer_qubits[x_stab_idx] = qubits
                    x_stab_idx += 1
        
        for col in range(0, d - 1, 2):
            self.x_stabilizer_positions[(-1, col)] = x_stab_idx
            qubits = [
                self.data_qubit_positions[(0, col)], 
                self.data_qubit_positions[(0, col + 1)]
            ]
            self.x_stabilizer_qubits[x_stab_idx] = qubits
            x_stab_idx += 1
        
        for col in range(1, d - 1, 2):
            self.x_stabilizer_positions[(d - 1, col)] = x_stab_idx
            qubits = [
                self.data_qubit_positions[(d - 1, col)], 
                self.data_qubit_positions[(d - 1, col + 1)]
            ]
            self.x_stabilizer_qubits[x_stab_idx] = qubits
            x_stab_idx += 1
        
        self.num_x_stabilizers = x_stab_idx
        
        z_stab_idx = 0
        for row in range(d - 1):
            for col in range(d - 1):
                if (row + col) % 2 == 1:
                    self.z_stabilizer_positions[(row, col)] = z_stab_idx
                    qubits = []
                    for dr, dc in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                        pos = (row + dr, col + dc)
                        if pos in self.data_qubit_positions:
                            qubits.append(self.data_qubit_positions[pos])
                    self.z_stabilizer_qubits[z_stab_idx] = qubits
                    z_stab_idx += 1
        
        for row in range(0, d - 1, 2):
            self.z_stabilizer_positions[(row, -1)] = z_stab_idx
            qubits = [
                self.data_qubit_positions[(row, 0)], 
                self.data_qubit_positions[(row + 1, 0)]
            ]
            self.z_stabilizer_qubits[z_stab_idx] = qubits
            z_stab_idx += 1
        
        for row in range(1, d - 1, 2):
            self.z_stabilizer_positions[(row, d - 1)] = z_stab_idx
            qubits = [
                self.data_qubit_positions[(row, d - 1)], 
                self.data_qubit_positions[(row + 1, d - 1)]
            ]
            self.z_stabilizer_qubits[z_stab_idx] = qubits
            z_stab_idx += 1
        
        self.num_z_stabilizers = z_stab_idx
    
    def measure_syndrome(self, errors: List[PauliOperator]) -> Tuple[np.ndarray, np.ndarray]:
        """Measure error syndrome from Pauli errors."""
        if len(errors) != self.num_data_qubits:
            raise ValueError(f"Expected {self.num_data_qubits} errors, got {len(errors)}")
        
        x_syndrome = np.zeros(self.num_x_stabilizers, dtype=np.int8)
        z_syndrome = np.zeros(self.num_z_stabilizers, dtype=np.int8)
        
        for stab_idx, qubits in self.x_stabilizer_qubits.items():
            parity = 0
            for q in qubits:
                if errors[q] in [PauliOperator.Z, PauliOperator.Y]:
                    parity ^= 1
            x_syndrome[stab_idx] = parity
        
        for stab_idx, qubits in self.z_stabilizer_qubits.items():
            parity = 0
            for q in qubits:
                if errors[q] in [PauliOperator.X, PauliOperator.Y]:
                    parity ^= 1
            z_syndrome[stab_idx] = parity
        
        return x_syndrome, z_syndrome
    
    def get_logical_x_operator(self) -> List[int]:
        """Get data qubits that form the logical X operator."""
        return [
            self.data_qubit_positions[(self.distance // 2, col)] 
            for col in range(self.distance)
        ]
    
    def get_logical_z_operator(self) -> List[int]:
        """Get data qubits that form the logical Z operator."""
        return [
            self.data_qubit_positions[(row, self.distance // 2)] 
            for row in range(self.distance)
        ]
    
    def __repr__(self) -> str:
        return (
            f"SurfaceCode(distance={self.distance}, "
            f"data_qubits={self.num_data_qubits}, "
            f"x_stabilizers={self.num_x_stabilizers}, "
            f"z_stabilizers={self.num_z_stabilizers})"
        )
