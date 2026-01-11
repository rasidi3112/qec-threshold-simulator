import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from qec_threshold.core.pauli import PauliOperator
from qec_threshold.core.surface_code import SurfaceCode


def test_pauli_multiplication():
    assert PauliOperator.I * PauliOperator.X == PauliOperator.X
    assert PauliOperator.X * PauliOperator.X == PauliOperator.I
    assert PauliOperator.X * PauliOperator.Z == PauliOperator.Y
    print("✓ Pauli multiplication tests passed")


def test_surface_code_initialization():
    for d in [3, 5, 7]:
        code = SurfaceCode(distance=d)
        assert code.distance == d
        assert code.num_data_qubits == d * d
        assert code.num_x_stabilizers > 0
        assert code.num_z_stabilizers > 0
    print("✓ Surface code initialization tests passed")


def test_syndrome_measurement():
    code = SurfaceCode(distance=3)
    errors = [PauliOperator.I] * code.num_data_qubits
    x_synd, z_synd = code.measure_syndrome(errors)
    assert np.sum(x_synd) == 0
    assert np.sum(z_synd) == 0
    print("✓ Syndrome measurement tests passed")


def test_logical_operators():
    code = SurfaceCode(distance=5)
    logical_x = code.get_logical_x_operator()
    logical_z = code.get_logical_z_operator()
    assert len(logical_x) == code.distance
    assert len(logical_z) == code.distance
    print("✓ Logical operator tests passed")


if __name__ == "__main__":
    test_pauli_multiplication()
    test_surface_code_initialization()
    test_syndrome_measurement()
    test_logical_operators()
    print("\n✅ All core tests passed!")
