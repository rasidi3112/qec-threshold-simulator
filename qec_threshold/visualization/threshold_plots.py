import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.patches import Circle, RegularPolygon

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.threshold_estimator import ThresholdResult
from core.surface_code import SurfaceCode
from core.pauli import PauliOperator
from noise_models.depolarizing import DepolarizingChannel


COLORS = {
    3: '#E63946',
    5: '#457B9D',
    7: '#2A9D8F',
    9: '#E9C46A',
    11: '#9B59B6',
    13: '#1ABC9C',
}


class ThresholdVisualizer:
    """Publication-quality visualization for threshold estimation results."""
    
    def __init__(self, result: ThresholdResult):
        self.result = result
        self.colors = COLORS
    
    def create_threshold_plot(self, save_path: str = None):
        """Create main threshold plot showing curves crossing at p_th."""
        plt.style.use('dark_background')
        
        fig, ax = plt.subplots(figsize=(14, 10), facecolor='#0a0a0f')
        ax.set_facecolor('#0a0a0f')
        ax.grid(True, alpha=0.15, linestyle='--', color='white')
        
        for d in self.result.distances_tested:
            color = self.colors.get(d, '#FFFFFF')
            rates = self.result.logical_error_rates[d]
            
            ax.semilogy(
                self.result.physical_error_rates * 100, rates,
                'o-', color=color, linewidth=2.5, markersize=8,
                label=f'd = {d}', alpha=0.9
            )
            
            ax.semilogy(
                self.result.physical_error_rates * 100, rates,
                '-', color=color, linewidth=6, alpha=0.3
            )
        
        p_th = self.result.estimated_threshold * 100
        ax.axvline(x=p_th, color='#FFD700', linestyle='--', linewidth=2, alpha=0.8)
        
        ax.annotate(
            f'Threshold\np_th ≈ {p_th:.2f}%',
            xy=(p_th, 0.1),
            xytext=(p_th + 0.3, 0.2),
            fontsize=14,
            color='#FFD700',
            fontweight='bold',
            ha='left',
            arrowprops=dict(arrowstyle='->', color='#FFD700', lw=2),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', 
                     edgecolor='#FFD700', alpha=0.9)
        )
        
        ax.text(
            p_th - 0.5, 0.001, 'QEC\nSucceeds', 
            fontsize=12, color='#2A9D8F', ha='right', va='center', 
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#0a0a0f', 
                     edgecolor='#2A9D8F', alpha=0.8)
        )
        
        ax.text(
            p_th + 0.5, 0.001, 'QEC\nFails', 
            fontsize=12, color='#E63946', ha='left', va='center', 
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#0a0a0f', 
                     edgecolor='#E63946', alpha=0.8)
        )
        
        ax.set_xlabel('Physical Error Rate (%)', fontsize=14, fontweight='bold', color='white')
        ax.set_ylabel('Logical Error Rate', fontsize=14, fontweight='bold', color='white')
        
        title = ax.set_title(
            'Surface Code Error Correction Threshold\n'
            'The Critical Point of Fault-Tolerant Quantum Computing',
            fontsize=18, fontweight='bold', color='white', pad=20
        )
        title.set_path_effects([path_effects.withStroke(linewidth=3, foreground='#1a1a2e')])
        
        legend = ax.legend(loc='upper left', fontsize=12, framealpha=0.9,
                          facecolor='#1a1a2e', edgecolor='#333366')
        for text in legend.get_texts():
            text.set_color('white')
        
        ax.set_xlim(0, max(self.result.physical_error_rates) * 100 * 1.05)
        ax.set_ylim(1e-4, 1)
        
        total_sims = (len(self.result.distances_tested) * 
                     len(self.result.physical_error_rates) * 
                     self.result.num_trials)
        stats_text = (
            f"Simulation Parameters:\n"
            f"• Distances: {self.result.distances_tested}\n"
            f"• Trials per point: {self.result.num_trials:,}\n"
            f"• Total simulations: {total_sims:,}\n"
            f"• Computation time: {self.result.computation_time:.1f}s"
        )
        
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', 
                        edgecolor='#333366', alpha=0.95),
               color='#aaaaaa', family='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, facecolor='#0a0a0f', 
                       edgecolor='none', bbox_inches='tight')
            print(f"Saved threshold plot to: {save_path}")
        
        plt.close()
        return fig, ax
    
    def create_lattice_visualization(self, distance: int = 5, 
                                     error_rate: float = 0.05,
                                     save_path: str = None):
        """Visualize the surface code lattice with errors and syndromes."""
        code = SurfaceCode(distance)
        channel = DepolarizingChannel(error_rate)
        rng = np.random.default_rng(42)
        errors = channel.apply(code.num_data_qubits, rng)
        x_syndrome, z_syndrome = code.measure_syndrome(errors)
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 12), facecolor='#0a0a0f')
        ax.set_facecolor('#0a0a0f')
        
        for (row, col), idx in code.data_qubit_positions.items():
            error = errors[idx]
            
            if error == PauliOperator.I:
                color, edge_color = '#2E4057', '#4A6FA5'
            elif error == PauliOperator.X:
                color, edge_color = '#E63946', '#FF6B6B'
            elif error == PauliOperator.Z:
                color, edge_color = '#457B9D', '#7FB3D5'
            else:
                color, edge_color = '#9B59B6', '#BB8FCE'
            
            circle = Circle((col, -row), 0.35, facecolor=color, edgecolor=edge_color, 
                           linewidth=2, zorder=3)
            ax.add_patch(circle)
            
            ax.text(col, -row, f'{idx}', ha='center', va='center',
                   fontsize=8, color='white', fontweight='bold', zorder=4)
        
        for (row, col), idx in code.x_stabilizer_positions.items():
            if 0 <= row < distance - 1 and 0 <= col < distance - 1:
                cx, cy = col + 0.5, -(row + 0.5)
                
                if x_syndrome[idx]:
                    color, size = '#FFD700', 0.3
                else:
                    color, size = '#2A9D8F', 0.2
                
                square = RegularPolygon((cx, cy), numVertices=4, radius=size,
                                        facecolor=color, edgecolor='white',
                                        linewidth=1, alpha=0.8, zorder=2)
                ax.add_patch(square)
        
        for (row, col), idx in code.z_stabilizer_positions.items():
            if 0 <= row < distance - 1 and 0 <= col < distance - 1:
                cx, cy = col + 0.5, -(row + 0.5)
                
                if z_syndrome[idx]:
                    color, size = '#FFD700', 0.3
                else:
                    color, size = '#E9C46A', 0.2
                
                hexagon = RegularPolygon((cx, cy), numVertices=6, radius=size,
                                         facecolor=color, edgecolor='white',
                                         linewidth=1, alpha=0.8, zorder=2)
                ax.add_patch(hexagon)
        
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E4057',
                      markersize=15, label='No Error'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#E63946',
                      markersize=15, label='X Error (bit-flip)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#457B9D',
                      markersize=15, label='Z Error (phase-flip)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#9B59B6',
                      markersize=15, label='Y Error (both)'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#FFD700',
                      markersize=12, label='Syndrome defect'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
                 facecolor='#1a1a2e', edgecolor='#333366')
        
        ax.set_title(f'Surface Code Lattice (d={distance})\n'
                    f'Physical Error Rate: {error_rate*100:.1f}%',
                    fontsize=16, fontweight='bold', color='white', pad=20)
        
        ax.set_xlim(-0.5, distance - 0.5)
        ax.set_ylim(-distance + 0.5, 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, facecolor='#0a0a0f',
                       edgecolor='none', bbox_inches='tight')
            print(f"Saved lattice plot to: {save_path}")
        
        plt.close()
        return fig, ax
    
    def create_scaling_analysis(self, save_path: str = None):
        """Create plot showing how logical error rate scales with distance."""
        from scipy.optimize import curve_fit
        
        plt.style.use('dark_background')
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='#0a0a0f')
        
        for ax in axes:
            ax.set_facecolor('#0a0a0f')
            ax.grid(True, alpha=0.15, linestyle='--', color='white')
        
        ax1 = axes[0]
        p_below = self.result.estimated_threshold * 0.5
        idx_below = np.argmin(np.abs(self.result.physical_error_rates - p_below))
        
        distances = np.array(self.result.distances_tested)
        ler_below = np.array([self.result.logical_error_rates[d][idx_below] for d in distances])
        
        ax1.semilogy(distances, ler_below, 'o-', color='#2A9D8F', 
                    linewidth=3, markersize=12, label='Simulated')
        
        try:
            def exp_decay(d, A, alpha):
                return A * np.exp(-alpha * d)
            popt, _ = curve_fit(exp_decay, distances, ler_below, p0=[1, 0.5])
            d_fit = np.linspace(min(distances), max(distances), 100)
            ax1.semilogy(d_fit, exp_decay(d_fit, *popt), '--', color='#E9C46A',
                        linewidth=2, label=f'Fit: A·exp(-{popt[1]:.2f}·d)')
        except Exception:
            pass
        
        ax1.set_xlabel('Code Distance (d)', fontsize=12, fontweight='bold', color='white')
        ax1.set_ylabel('Logical Error Rate', fontsize=12, fontweight='bold', color='white')
        ax1.set_title(f'Below Threshold (p = {p_below*100:.2f}%)\nExponential Suppression',
                     fontsize=14, fontweight='bold', color='#2A9D8F')
        ax1.legend(fontsize=10, facecolor='#1a1a2e', edgecolor='#333366')
        
        ax2 = axes[1]
        p_above = self.result.estimated_threshold * 1.5
        idx_above = np.argmin(np.abs(self.result.physical_error_rates - p_above))
        
        ler_above = np.array([self.result.logical_error_rates[d][idx_above] for d in distances])
        
        ax2.plot(distances, ler_above, 'o-', color='#E63946', linewidth=3, markersize=12)
        
        ax2.set_xlabel('Code Distance (d)', fontsize=12, fontweight='bold', color='white')
        ax2.set_ylabel('Logical Error Rate', fontsize=12, fontweight='bold', color='white')
        ax2.set_title(f'Above Threshold (p = {p_above*100:.2f}%)\nError Accumulation',
                     fontsize=14, fontweight='bold', color='#E63946')
        
        fig.suptitle('The Threshold Theorem in Action\n'
                    'Below threshold: errors are exponentially suppressed\n'
                    'Above threshold: more qubits = more errors!',
                    fontsize=16, fontweight='bold', color='white', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, facecolor='#0a0a0f',
                       edgecolor='none', bbox_inches='tight')
            print(f"Saved scaling plot to: {save_path}")
        
        plt.close()
        return fig, axes
    
    def __repr__(self) -> str:
        return f"ThresholdVisualizer(distances={self.result.distances_tested})"
