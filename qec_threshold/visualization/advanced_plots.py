import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.threshold_estimator import ThresholdResult
from analysis.finite_size_scaling import FiniteSizeScalingResult


COLORS = {
    3: '#FF6B6B',
    5: '#4ECDC4',
    7: '#45B7D1',
    9: '#96CEB4',
    11: '#FFEAA7',
    13: '#DDA0DD',
}


class AdvancedVisualizer:
    """Advanced visualizations for deep QEC analysis."""
    
    def __init__(self, result: ThresholdResult):
        self.result = result
        self.colors = COLORS
    
    def plot_scaling_collapse(self, fss_result: FiniteSizeScalingResult,
                              save_path: str = None):
        """Plot finite-size scaling collapse."""
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#0a0a0f')
        
        for ax in [ax1, ax2]:
            ax.set_facecolor('#0a0a0f')
            ax.grid(True, alpha=0.15, linestyle='--', color='white')
        
        for d in self.result.distances_tested:
            color = self.colors.get(d, '#FFFFFF')
            p_L = self.result.logical_error_rates[d]
            valid = p_L > 0
            
            ax1.semilogy(self.result.physical_error_rates[valid] * 100, p_L[valid],
                        'o-', color=color, linewidth=2, markersize=8, label=f'd = {d}')
        
        ax1.axvline(x=fss_result.threshold * 100, color='#FFD700', linestyle='--', linewidth=2)
        ax1.set_xlabel('Physical Error Rate (%)', fontsize=12, color='white')
        ax1.set_ylabel('Logical Error Rate', fontsize=12, color='white')
        ax1.set_title('Raw Data', fontsize=14, fontweight='bold', color='white')
        ax1.legend(fontsize=10, facecolor='#1a1a2e', edgecolor='#333366')
        
        for d, (x, y) in fss_result.collapsed_data.items():
            color = self.colors.get(d, '#FFFFFF')
            ax2.semilogy(x, y, 'o-', color=color, linewidth=2, markersize=8, label=f'd = {d}')
        
        ax2.axvline(x=0, color='#FFD700', linestyle='--', linewidth=2)
        ax2.set_xlabel(r'$(p - p_{th}) \cdot d^{1/\nu}$', fontsize=12, color='white')
        ax2.set_ylabel(r'$p_L$', fontsize=12, color='white')
        ax2.set_title(f'Scaling Collapse (ν = {fss_result.critical_exponent_nu:.2f})',
                     fontsize=14, fontweight='bold', color='white')
        ax2.legend(fontsize=10, facecolor='#1a1a2e', edgecolor='#333366')
        
        info_text = (f"Threshold: p_th = {fss_result.threshold*100:.2f}%\n"
                    f"Critical exponent: ν = {fss_result.critical_exponent_nu:.2f}\n"
                    f"Scaling quality: {fss_result.scaling_quality:.3f}")
        
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=11,
                color='#aaaaaa', family='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e',
                         edgecolor='#FFD700', alpha=0.9))
        
        fig.suptitle('Finite-Size Scaling Analysis\n'
                    'Universality Class of the QEC Phase Transition',
                    fontsize=16, fontweight='bold', color='white', y=0.98)
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, facecolor='#0a0a0f',
                       edgecolor='none', bbox_inches='tight')
            print(f"Saved scaling collapse plot to: {save_path}")
        
        plt.close()
        return fig
    
    def plot_decoder_comparison(self, benchmark_results: Dict[float, Dict[str, float]],
                                save_path: str = None):
        """Plot decoder performance comparison."""
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='#0a0a0f')
        ax.set_facecolor('#0a0a0f')
        ax.grid(True, alpha=0.15, linestyle='--', color='white')
        
        error_rates = sorted(benchmark_results.keys())
        
        decoder_colors = {'mwpm': '#4ECDC4', 'bp': '#FF6B6B', 'union_find': '#FFEAA7'}
        decoder_labels = {'mwpm': 'MWPM (Gold Standard)', 'bp': 'Belief Propagation', 
                         'union_find': 'Union-Find'}
        
        for decoder in decoder_colors:
            if decoder in benchmark_results[error_rates[0]]:
                rates = [benchmark_results[p].get(decoder, 0) for p in error_rates]
                color = decoder_colors[decoder]
                label = decoder_labels.get(decoder, decoder)
                
                ax.semilogy([p * 100 for p in error_rates], rates,
                           'o-', color=color, linewidth=2.5, markersize=10, label=label)
        
        ax.set_xlabel('Physical Error Rate (%)', fontsize=14, fontweight='bold', color='white')
        ax.set_ylabel('Logical Error Rate', fontsize=14, fontweight='bold', color='white')
        ax.set_title('Decoder Performance Comparison\nTrade-off: Speed vs Accuracy',
                    fontsize=16, fontweight='bold', color='white', pad=20)
        ax.legend(fontsize=12, facecolor='#1a1a2e', edgecolor='#333366')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, facecolor='#0a0a0f',
                       edgecolor='none', bbox_inches='tight')
            print(f"Saved decoder comparison to: {save_path}")
        
        plt.close()
        return fig
    
    def plot_biased_noise_analysis(self, bias_results: Dict[float, float],
                                   save_path: str = None):
        """Plot threshold vs noise bias."""
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='#0a0a0f')
        ax.set_facecolor('#0a0a0f')
        ax.grid(True, alpha=0.15, linestyle='--', color='white')
        
        biases = sorted(bias_results.keys())
        thresholds = [bias_results[b] * 100 for b in biases]
        
        ax.semilogx(biases, thresholds, 'o-', color='#FF6B6B', linewidth=3, markersize=12)
        ax.axhline(y=1.0, color='#FFD700', linestyle='--', linewidth=2, 
                  label='Depolarizing threshold (~1%)')
        ax.fill_between(biases, 0, thresholds, alpha=0.3, color='#FF6B6B')
        
        ax.set_xlabel('Bias Ratio η = P(Z)/P(X)', fontsize=14, fontweight='bold', color='white')
        ax.set_ylabel('Threshold (%)', fontsize=14, fontweight='bold', color='white')
        ax.set_title('Threshold Improvement with Biased Noise\n'
                    'Highly Biased Noise Enables Higher Thresholds',
                    fontsize=16, fontweight='bold', color='white', pad=20)
        ax.legend(fontsize=12, facecolor='#1a1a2e', edgecolor='#333366')
        
        if 100 in bias_results:
            ax.annotate('Superconducting qubits\noften have η ~ 100',
                       xy=(100, bias_results[100] * 100),
                       xytext=(30, 8),
                       fontsize=10, color='#aaaaaa',
                       arrowprops=dict(arrowstyle='->', color='#aaaaaa'),
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', 
                               edgecolor='#333366', alpha=0.9))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, facecolor='#0a0a0f',
                       edgecolor='none', bbox_inches='tight')
            print(f"Saved biased noise analysis to: {save_path}")
        
        plt.close()
        return fig
    
    def __repr__(self) -> str:
        return f"AdvancedVisualizer(distances={self.result.distances_tested})"
