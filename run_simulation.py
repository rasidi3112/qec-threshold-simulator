#!/usr/bin/env python3
"""Quantum Error Correction Threshold Simulator - Main Entry Point."""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qec_threshold import (
    ThresholdEstimator,
    ThresholdVisualizer,
    FiniteSizeScalingAnalyzer,
    AdvancedVisualizer,
)


def print_banner():
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║        QUANTUM ERROR CORRECTION THRESHOLD SIMULATOR                              ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝
    """)


def main():
    parser = argparse.ArgumentParser(description='Quantum Error Correction Threshold Simulator')
    parser.add_argument('--distances', type=int, nargs='+', default=[3, 5, 7])
    parser.add_argument('--trials', type=int, default=500)
    parser.add_argument('--output', type=str, default='outputs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--skip-plots', action='store_true')
    
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    print_banner()
    
    print("\n[1/4] Configuring simulation parameters...")
    
    distances = sorted(args.distances)
    error_rates = np.concatenate([
        np.linspace(0.002, 0.008, 4),
        np.linspace(0.009, 0.015, 7),
        np.linspace(0.016, 0.022, 4)
    ])
    
    print(f"    • Code distances: {distances}")
    print(f"    • Error rate range: {error_rates[0]*100:.2f}% to {error_rates[-1]*100:.2f}%")
    print(f"    • Monte Carlo trials per point: {args.trials}")
    print(f"    • Total simulations: {len(distances) * len(error_rates) * args.trials:,}")
    
    print("\n[2/4] Initializing threshold estimator...")
    
    estimator = ThresholdEstimator(
        distances=distances,
        error_rates=error_rates,
        num_trials=args.trials,
        seed=args.seed
    )
    
    print("\n[3/4] Running Monte Carlo threshold estimation...")
    
    result = estimator.run_threshold_estimation(verbose=args.verbose)
    
    if not args.skip_plots:
        print("\n[4/4] Generating visualizations...")
        
        visualizer = ThresholdVisualizer(result)
        visualizer.create_threshold_plot(save_path=os.path.join(args.output, 'threshold_plot.png'))
        visualizer.create_lattice_visualization(
            distance=min(7, max(distances)),
            error_rate=0.08,
            save_path=os.path.join(args.output, 'lattice_visualization.png')
        )
        visualizer.create_scaling_analysis(save_path=os.path.join(args.output, 'scaling_analysis.png'))
        
        fss_analyzer = FiniteSizeScalingAnalyzer(result)
        fss_result = fss_analyzer.analyze()
        
        adv_visualizer = AdvancedVisualizer(result)
        adv_visualizer.plot_scaling_collapse(fss_result, save_path=os.path.join(args.output, 'scaling_collapse.png'))
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║                          SIMULATION COMPLETE                                     ║
    ║   Estimated threshold: p_th = {result.estimated_threshold*100:.2f}%                                        ║
    ║   Computation time: {result.computation_time:.1f}s                                                   ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    return result


if __name__ == "__main__":
    result = main()
