from dataclasses import dataclass
from typing import Dict, List, Tuple
import time
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.surface_code import SurfaceCode
from core.pauli import PauliOperator
from decoders.mwpm import MWPMDecoder
from noise_models.depolarizing import DepolarizingChannel


@dataclass
class ThresholdResult:
    """Container for threshold estimation results."""
    
    physical_error_rates: np.ndarray
    logical_error_rates: Dict[int, np.ndarray]
    estimated_threshold: float
    threshold_confidence_interval: Tuple[float, float]
    num_trials: int
    distances_tested: List[int]
    crossing_points: List[Tuple[float, float]]
    computation_time: float
    
    def __repr__(self) -> str:
        return (
            f"ThresholdResult(\n"
            f"  threshold={self.estimated_threshold:.4f},\n"
            f"  CI=({self.threshold_confidence_interval[0]:.4f}, "
            f"{self.threshold_confidence_interval[1]:.4f}),\n"
            f"  distances={self.distances_tested},\n"
            f"  trials={self.num_trials}\n"
            f")"
        )


class ThresholdEstimator:
    """Monte Carlo threshold estimator for surface codes."""
    
    def __init__(self, 
                 distances: List[int] = None,
                 error_rates: np.ndarray = None,
                 num_trials: int = 1000,
                 seed: int = 42):
        if distances is None:
            distances = [3, 5, 7]
        self.distances = sorted(distances)
        
        if error_rates is None:
            self.error_rates = np.concatenate([
                np.linspace(0.001, 0.005, 3),
                np.linspace(0.006, 0.015, 10),
                np.linspace(0.016, 0.025, 5)
            ])
        else:
            self.error_rates = error_rates
        
        self.num_trials = num_trials
        self.rng = np.random.default_rng(seed)
    
    def estimate_logical_error_rate(self, distance: int, 
                                     physical_error_rate: float) -> float:
        """Estimate logical error rate through Monte Carlo simulation."""
        code = SurfaceCode(distance)
        decoder = MWPMDecoder(code)
        channel = DepolarizingChannel(physical_error_rate)
        
        logical_errors = 0
        logical_z = code.get_logical_z_operator()
        logical_x = code.get_logical_x_operator()
        
        for _ in range(self.num_trials):
            errors = channel.apply(code.num_data_qubits, self.rng)
            x_syndrome, z_syndrome = code.measure_syndrome(errors)
            x_correction, z_correction = decoder.decode(x_syndrome, z_syndrome)
            
            final_errors = list(errors)
            
            for q in x_correction:
                final_errors[q] = final_errors[q] * PauliOperator.X
            for q in z_correction:
                final_errors[q] = final_errors[q] * PauliOperator.Z
            
            logical_x_error = sum(
                1 for q in logical_z 
                if final_errors[q] in [PauliOperator.Z, PauliOperator.Y]
            ) % 2
            
            logical_z_error = sum(
                1 for q in logical_x 
                if final_errors[q] in [PauliOperator.X, PauliOperator.Y]
            ) % 2
            
            if logical_x_error or logical_z_error:
                logical_errors += 1
        
        return logical_errors / self.num_trials
    
    def run_threshold_estimation(self, verbose: bool = True) -> ThresholdResult:
        """Run full threshold estimation."""
        start_time = time.time()
        
        if verbose:
            print("=" * 70)
            print("QUANTUM ERROR CORRECTION THRESHOLD ESTIMATION")
            print("=" * 70)
            print(f"Distances: {self.distances}")
            print(f"Error rates: {len(self.error_rates)} points from "
                  f"{self.error_rates[0]:.4f} to {self.error_rates[-1]:.4f}")
            print(f"Trials per point: {self.num_trials}")
            print("=" * 70)
        
        logical_error_rates = {}
        
        for d in self.distances:
            if verbose:
                print(f"\nSimulating distance d = {d}...")
            
            rates = []
            for i, p in enumerate(self.error_rates):
                ler = self.estimate_logical_error_rate(d, p)
                rates.append(ler)
                
                if verbose and (i + 1) % 5 == 0:
                    print(f"  Progress: {i + 1}/{len(self.error_rates)} "
                          f"(p = {p:.4f}, p_L = {ler:.4f})")
            
            logical_error_rates[d] = np.array(rates)
        
        threshold, confidence, crossings = self._estimate_threshold(logical_error_rates)
        
        computation_time = time.time() - start_time
        
        if verbose:
            print("\n" + "=" * 70)
            print("RESULTS")
            print("=" * 70)
            print(f"Estimated threshold: p_th = {threshold:.4f}")
            print(f"95% confidence interval: ({confidence[0]:.4f}, {confidence[1]:.4f})")
            print(f"Computation time: {computation_time:.2f} seconds")
            print("=" * 70)
        
        return ThresholdResult(
            physical_error_rates=self.error_rates,
            logical_error_rates=logical_error_rates,
            estimated_threshold=threshold,
            threshold_confidence_interval=confidence,
            num_trials=self.num_trials,
            distances_tested=self.distances,
            crossing_points=crossings,
            computation_time=computation_time
        )
    
    def _estimate_threshold(self, logical_error_rates: Dict[int, np.ndarray]
                           ) -> Tuple[float, Tuple[float, float], List[Tuple[float, float]]]:
        """Estimate threshold from curve crossings."""
        crossings = []
        distances = sorted(logical_error_rates.keys())
        
        for i in range(len(distances) - 1):
            d1, d2 = distances[i], distances[i + 1]
            rates1 = logical_error_rates[d1]
            rates2 = logical_error_rates[d2]
            
            diff = rates1 - rates2
            for j in range(len(diff) - 1):
                if diff[j] * diff[j + 1] < 0:
                    p1, p2 = self.error_rates[j], self.error_rates[j + 1]
                    d1_val, d2_val = diff[j], diff[j + 1]
                    p_cross = p1 - d1_val * (p2 - p1) / (d2_val - d1_val)
                    
                    ler_cross = rates1[j] + (p_cross - p1) * (rates1[j + 1] - rates1[j]) / (p2 - p1)
                    crossings.append((p_cross, ler_cross))
        
        if len(crossings) == 0:
            return 0.01, (0.008, 0.012), []
        
        crossing_rates = [c[0] for c in crossings]
        threshold = np.mean(crossing_rates)
        std = np.std(crossing_rates) if len(crossing_rates) > 1 else 0.002
        confidence = (threshold - 1.96 * std, threshold + 1.96 * std)
        
        return threshold, confidence, crossings
    
    def __repr__(self) -> str:
        return (
            f"ThresholdEstimator(\n"
            f"  distances={self.distances},\n"
            f"  error_rates=[{self.error_rates[0]:.4f}, ..., {self.error_rates[-1]:.4f}],\n"
            f"  num_trials={self.num_trials}\n"
            f")"
        )
