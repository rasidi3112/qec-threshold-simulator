from typing import Dict, List, Tuple
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.surface_code import SurfaceCode


class UnionFind:
    """Union-Find (Disjoint Set Union) data structure."""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        """Find the representative of the set containing x."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """Merge the sets containing x and y."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        
        return True


class MWPMDecoder:
    """Minimum Weight Perfect Matching Decoder for Surface Codes."""
    
    def __init__(self, surface_code: SurfaceCode):
        self.code = surface_code
        self.x_stab_coords: Dict[int, Tuple[float, float]] = {}
        self.z_stab_coords: Dict[int, Tuple[float, float]] = {}
        self._precompute_distances()
    
    def _precompute_distances(self):
        for (row, col), idx in self.code.x_stabilizer_positions.items():
            self.x_stab_coords[idx] = (row + 0.5, col + 0.5)
        
        for (row, col), idx in self.code.z_stabilizer_positions.items():
            self.z_stab_coords[idx] = (row + 0.5, col + 0.5)
    
    def _manhattan_distance(self, coord1: Tuple[float, float], 
                           coord2: Tuple[float, float]) -> float:
        return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])
    
    def _greedy_matching(self, defects: List[int], 
                         coords: Dict[int, Tuple[float, float]],
                         boundary_coord: Tuple[float, float]) -> List[Tuple[int, int]]:
        """Greedy approximation to minimum weight perfect matching."""
        if len(defects) == 0:
            return []
        
        all_nodes = list(defects)
        is_boundary = {d: False for d in defects}
        
        if len(defects) % 2 == 1:
            boundary_id = -1
            all_nodes.append(boundary_id)
            is_boundary[boundary_id] = True
        
        n = len(all_nodes)
        node_to_idx = {node: i for i, node in enumerate(all_nodes)}
        
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                node_i, node_j = all_nodes[i], all_nodes[j]
                
                coord_i = boundary_coord if is_boundary.get(node_i, False) else coords[node_i]
                coord_j = boundary_coord if is_boundary.get(node_j, False) else coords[node_j]
                
                dist = self._manhattan_distance(coord_i, coord_j)
                edges.append((dist, i, j))
        
        edges.sort()
        uf = UnionFind(n)
        degree = [0] * n
        matching = []
        
        for dist, i, j in edges:
            if degree[i] == 0 and degree[j] == 0:
                if uf.find(i) != uf.find(j) or len(matching) == n // 2 - 1:
                    matching.append((all_nodes[i], all_nodes[j]))
                    degree[i] = 1
                    degree[j] = 1
                    uf.union(i, j)
        
        return matching
    
    def decode(self, x_syndrome: np.ndarray, 
               z_syndrome: np.ndarray) -> Tuple[List[int], List[int]]:
        """Decode syndrome to find correction operator."""
        x_defects = np.where(x_syndrome == 1)[0].tolist()
        z_defects = np.where(z_syndrome == 1)[0].tolist()
        
        x_boundary = (self.code.distance / 2, -0.5)
        x_matching = self._greedy_matching(x_defects, self.x_stab_coords, x_boundary)
        
        z_boundary = (-0.5, self.code.distance / 2)
        z_matching = self._greedy_matching(z_defects, self.z_stab_coords, z_boundary)
        
        z_correction = self._matching_to_correction(
            x_matching, self.x_stab_coords, x_boundary, 'z'
        )
        x_correction = self._matching_to_correction(
            z_matching, self.z_stab_coords, z_boundary, 'x'
        )
        
        return x_correction, z_correction
    
    def _matching_to_correction(self, matching: List[Tuple[int, int]],
                                 coords: Dict[int, Tuple[float, float]],
                                 boundary_coord: Tuple[float, float],
                                 correction_type: str) -> List[int]:
        """Convert matching to list of qubits needing correction."""
        corrections = set()
        
        for node1, node2 in matching:
            coord1 = boundary_coord if node1 == -1 else coords[node1]
            coord2 = boundary_coord if node2 == -1 else coords[node2]
            
            path_qubits = self._find_path_qubits(coord1, coord2, correction_type)
            
            for q in path_qubits:
                if q in corrections:
                    corrections.remove(q)
                else:
                    corrections.add(q)
        
        return list(corrections)
    
    def _find_path_qubits(self, coord1: Tuple[float, float], 
                          coord2: Tuple[float, float],
                          correction_type: str) -> List[int]:
        """Find data qubits along shortest path between two coordinates."""
        qubits = []
        d = self.code.distance
        
        r1, c1 = coord1
        r2, c2 = coord2
        
        current_r, current_c = r1, c1
        
        step_c = 1 if c2 > c1 else -1
        while abs(current_c - c2) > 0.5:
            qr = int(round(current_r - 0.5)) if correction_type == 'z' else int(round(current_r))
            qc = int(current_c) if step_c > 0 else int(current_c - 1)
            
            qr = max(0, min(d - 1, qr))
            qc = max(0, min(d - 1, qc))
            
            if (qr, qc) in self.code.data_qubit_positions:
                qubits.append(self.code.data_qubit_positions[(qr, qc)])
            current_c += step_c
        
        step_r = 1 if r2 > r1 else -1
        while abs(current_r - r2) > 0.5:
            qr = int(current_r) if step_r > 0 else int(current_r - 1)
            qc = int(round(current_c - 0.5)) if correction_type == 'x' else int(round(current_c))
            
            qr = max(0, min(d - 1, qr))
            qc = max(0, min(d - 1, qc))
            
            if (qr, qc) in self.code.data_qubit_positions:
                qubits.append(self.code.data_qubit_positions[(qr, qc)])
            current_r += step_r
        
        return qubits
    
    def __repr__(self) -> str:
        return f"MWPMDecoder(code_distance={self.code.distance})"
