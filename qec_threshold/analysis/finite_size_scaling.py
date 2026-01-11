from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from scipy.optimize import minimize

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.threshold_estimator import ThresholdResult


@dataclass
class FiniteSizeScalingResult:
    """Results from finite-size scaling analysis."""
    
    threshold: float
    critical_exponent_nu: float
    scaling_quality: float
    collapsed_data: Dict[int, Tuple[np.ndarray, np.ndarray]]
    
    def __repr__(self) -> str:
        return (
            f"FiniteSizeScalingResult(\n"
            f"  threshold={self.threshold:.4f},\n"
            f"  nu={self.critical_exponent_nu:.2f},\n"
            f"  quality={self.scaling_quality:.3f}\n"
            f")"
        )


class FiniteSizeScalingAnalyzer:
    """Finite-size scaling analysis for QEC threshold."""
    
    def __init__(self, result: ThresholdResult):
        self.result = result
    
    def analyze(self, nu_initial: float = 1.3, 
                beta_over_nu: float = 0.0) -> FiniteSizeScalingResult:
        """Perform finite-size scaling analysis."""
        
        def collapse_quality(params):
            p_th, nu = params
            
            all_scaled_x = []
            all_scaled_y = []
            
            for d in self.result.distances_tested:
                p_arr = self.result.physical_error_rates
                p_L_arr = self.result.logical_error_rates[d]
                
                valid = p_L_arr > 0
                if not np.any(valid):
                    continue
                
                p_valid = p_arr[valid]
                p_L_valid = p_L_arr[valid]
                
                x_scaled = (p_valid - p_th) * (d ** (1 / nu))
                y_scaled = p_L_valid * (d ** beta_over_nu)
                
                all_scaled_x.extend(x_scaled)
                all_scaled_y.extend(y_scaled)
            
            if len(all_scaled_x) < 2:
                return 1e10
            
            sorted_indices = np.argsort(all_scaled_x)
            x_sorted = np.array(all_scaled_x)[sorted_indices]
            y_sorted = np.array(all_scaled_y)[sorted_indices]
            
            window = max(3, len(x_sorted) // 10)
            variance = 0
            count = 0
            
            for i in range(len(x_sorted) - window):
                local_y = y_sorted[i:i + window]
                variance += np.var(local_y)
                count += 1
            
            return variance / max(count, 1)
        
        initial_guess = [self.result.estimated_threshold, nu_initial]
        bounds = [(0.005, 0.02), (0.5, 3.0)]
        
        opt_result = minimize(
            collapse_quality, 
            initial_guess, 
            method='L-BFGS-B', 
            bounds=bounds
        )
        
        optimal_threshold, optimal_nu = opt_result.x
        
        collapsed_data = {}
        for d in self.result.distances_tested:
            p_arr = self.result.physical_error_rates
            p_L_arr = self.result.logical_error_rates[d]
            
            valid = p_L_arr > 0
            if np.any(valid):
                p_valid = p_arr[valid]
                p_L_valid = p_L_arr[valid]
                
                x_scaled = (p_valid - optimal_threshold) * (d ** (1 / optimal_nu))
                y_scaled = p_L_valid * (d ** beta_over_nu)
                
                collapsed_data[d] = (x_scaled, y_scaled)
        
        quality = 1.0 / (1.0 + opt_result.fun)
        
        return FiniteSizeScalingResult(
            threshold=optimal_threshold,
            critical_exponent_nu=optimal_nu,
            scaling_quality=quality,
            collapsed_data=collapsed_data
        )
    
    def __repr__(self) -> str:
        return f"FiniteSizeScalingAnalyzer(distances={self.result.distances_tested})"
