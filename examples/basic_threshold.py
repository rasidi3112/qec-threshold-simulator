"""Basic Threshold Estimation Example."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qec_threshold import SurfaceCode, ThresholdEstimator, ThresholdVisualizer


def main():
    print("Basic Threshold Estimation Example")
    print("=" * 40)
    
    code = SurfaceCode(distance=5)
    print(f"Created: {code}")
    
    estimator = ThresholdEstimator(distances=[3, 5, 7], num_trials=200, seed=42)
    result = estimator.run_threshold_estimation(verbose=False)
    
    print(f"\nEstimated threshold: {result.estimated_threshold*100:.2f}%")
    print(f"Computation time: {result.computation_time:.2f}s")
    
    visualizer = ThresholdVisualizer(result)
    visualizer.create_threshold_plot(save_path='example_threshold.png')
    
    print("\nDone! Check 'example_threshold.png'")


if __name__ == "__main__":
    main()
