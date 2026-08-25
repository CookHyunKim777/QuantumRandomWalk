"""
Utilities for quantum--random hybrid walks (QRWs).
- Graph construction, walk simulation, and measurement functions
- Each graph constructor includes dist_sq, the squared distance from the start
- Walk functions calculate the MSD on the fly and save snapshots at save_interval
"""

import numpy as np
import scipy as sc
import networkx as nx
import hiperwalk as hpw


# ============================================================
# Basic matrix and probability functions
# ============================================================

def uniform_matrix(n):
    """Return an n x n uniform matrix with entries 1/n."""
    return np.full((n, n), 1/n)


def adjacency_matrix(qw):
    """Construct the block-diagonal transition matrix for a coined QW."""
    graph = qw._graph
    num_vert = graph.number_of_vertices()
    degree = graph.degree
    blocks = [uniform_matrix(degree(v)) for v in range(num_vert)]
    C = sc.sparse.block_diag(blocks, format='csr')
    return sc.sparse.csr_array(C)


def single_prob(qw, state_diag):
    """Calculate vertex probabilities from diagonal state entries."""
    graph = qw._graph
    num_vert = graph.number_of_vertices()
    return np.array([
        np.abs(state_diag[graph.arcs_with_tail(v)]).sum()
        for v in range(num_vert)
    ])


def probability_distribution(qw, states):
    """Calculate vertex probabilities from one or more state arrays."""
    states = np.asarray(states)
    if len(states.shape) == 1:
        states = np.asarray([states])

    graph = qw._graph
    num_vert = graph.number_of_vertices()

    prob = np.array([
        [np.abs(states[i, graph.arcs_with_tail(v)]).sum() for v in range(num_vert)]
        for i in range(len(states))
    ])
    return prob


# ============================================================
# Squared-distance arrays
# ============================================================

def _dist_sq_cycle(N, center):
    """Cycle: minimum image convention"""
    dist_sq = np.zeros(N)
    for v in range(N):
        dx = abs(v - center)
        dx = min(dx, N - dx)
        dist_sq[v] = dx ** 2
    return dist_sq


def _dist_sq_line(N, center):
    """Line: open boundary"""
    return np.array([(v - center) ** 2 for v in range(N)], dtype=np.float64)


def _dist_sq_periodic_2d(N, dim, center_coord, mapping):
    """Use the minimum-image convention on a periodic 2D lattice."""
    dist_sq = np.zeros(N, dtype=np.float64)
    reverse_mapping = {i: node for node, i in mapping.items()}
    cx, cy = center_coord
    for v in range(N):
        coord = reverse_mapping[v]
        dx = abs(coord[0] - cx)
        dy = abs(coord[1] - cy)
        dx = min(dx, dim - dx)
        dy = min(dy, dim - dy)
        dist_sq[v] = dx ** 2 + dy ** 2
    return dist_sq


def _dist_sq_periodic_3d(N, dim, center_coord, mapping):
    """Use the minimum-image convention on a periodic 3D lattice."""
    dist_sq = np.zeros(N, dtype=np.float64)
    reverse_mapping = {i: node for node, i in mapping.items()}
    cx, cy, cz = center_coord
    for v in range(N):
        coord = reverse_mapping[v]
        dx = abs(coord[0] - cx)
        dy = abs(coord[1] - cy)
        dz = abs(coord[2] - cz)
        dx = min(dx, dim - dx)
        dy = min(dy, dim - dy)
        dz = min(dz, dim - dz)
        dist_sq[v] = dx ** 2 + dy ** 2 + dz ** 2
    return dist_sq


def _dist_sq_bcc(N, dim, center_node, coord_to_node):
    """BCC lattice containing body-center and corner sites."""
    node_to_coord = {v: c for c, v in coord_to_node.items()}
    center_coord = node_to_coord[center_node]
    # BCC coordinates: (i, j, k, sublattice)
    # Real-space coordinates: corner=(i,j,k), body-center=(i+0.5,j+0.5,k+0.5)
    cx, cy, cz, cs = center_coord
    if cs == 1:  # body-center
        cx_real, cy_real, cz_real = cx + 0.5, cy + 0.5, cz + 0.5
    else:
        cx_real, cy_real, cz_real = float(cx), float(cy), float(cz)
    
    dist_sq = np.zeros(N, dtype=np.float64)
    for v in range(N):
        i, j, k, s = node_to_coord[v]
        if s == 1:
            x, y, z = i + 0.5, j + 0.5, k + 0.5
        else:
            x, y, z = float(i), float(j), float(k)
        
        dx = abs(x - cx_real)
        dy = abs(y - cy_real)
        dz = abs(z - cz_real)
        dx = min(dx, dim - dx)
        dy = min(dy, dim - dy)
        dz = min(dz, dim - dz)
        dist_sq[v] = dx ** 2 + dy ** 2 + dz ** 2
    return dist_sq


def _dist_sq_fcc(N, dim, center_coord, coord_to_node):
    """Use the minimum-image convention in integer coordinates on an FCC lattice."""
    node_to_coord = {v: c for c, v in coord_to_node.items()}
    cx, cy, cz = center_coord
    size = 2 * dim
    
    dist_sq = np.zeros(N, dtype=np.float64)
    for v in range(N):
        i, j, k = node_to_coord[v]
        dx = abs(i - cx)
        dy = abs(j - cy)
        dz = abs(k - cz)
        dx = min(dx, size - dx)
        dy = min(dy, size - dy)
        dz = min(dz, size - dz)
        # Real-space distance = integer-coordinate distance x 0.5 (FCC scaling).
        # Only relative distances matter here, so retain the integer coordinates.
        dist_sq[v] = dx ** 2 + dy ** 2 + dz ** 2
    return dist_sq


def _dist_sq_bfs(nx_graph, center):
    """Calculate graph distances with BFS (Bethe, (u,v)-flower, etc.)."""
    N = nx_graph.number_of_nodes()
    lengths = nx.single_source_shortest_path_length(nx_graph, center)
    dist_sq = np.zeros(N, dtype=np.float64)
    for v, d in lengths.items():
        dist_sq[v] = d ** 2
    return dist_sq

def _dist_sq_triangular_euclidean(mapping, center_int, dim):
    """Squared Euclidean distance on a triangular lattice with minimum images."""
    reverse_mapping = {v: node for node, v in mapping.items()}
    
    def pos(node):
        i, j = node
        return i + 0.5 * (j % 2), j * np.sqrt(3) / 2

    i_vals = [node[0] for node in mapping.keys()]
    j_vals = [node[1] for node in mapping.keys()]
    Lx = max(i_vals) + 1
    Ly = (max(j_vals) + 1) * np.sqrt(3) / 2

    cx, cy = pos(reverse_mapping[center_int])

    N = len(mapping)
    dist_sq = np.zeros(N, dtype=np.float64)
    for node, v in mapping.items():
        x, y = pos(node)
        dx = x - cx;  dx -= Lx * round(dx / Lx)
        dy = y - cy;  dy -= Ly * round(dy / Ly)
        dist_sq[v] = dx**2 + dy**2
    return dist_sq


def _dist_sq_honeycomb_euclidean(mapping, center_int, m, n_col):
    """Use the real-space coordinates of NetworkX hexagonal_lattice_graph.
    Periodic boundaries are handled with the minimum-image convention.
    """
    reverse_mapping = {v: node for node, v in mapping.items()}

    def pos(node):
        i, j = node
        # Follow the NetworkX coordinates to preserve regular hexagons
        x = 0.5 + i + i // 2 + (j % 2) * ((i % 2) - 0.5)
        y = j * np.sqrt(3) / 2
        return x, y

    cx, cy = pos(reverse_mapping[center_int])
    Lx = 1.5 * n_col          # Physical x period (n_col = columns of hexagons)
    Ly = m * np.sqrt(3)       # Physical y period (m = rows of hexagons)

    N = len(mapping)
    dist_sq = np.zeros(N, dtype=np.float64)
    for node, v in mapping.items():
        x, y = pos(node)
        dx = x - cx;  dx -= Lx * round(dx / Lx)
        dy = y - cy;  dy -= Ly * round(dy / Ly)
        dist_sq[v] = dx**2 + dy**2
    return dist_sq


# ============================================================
# Graph constructors
# ============================================================


def create_line_graph(dim, coin_type='H'):
    """Construct a one-dimensional line with open boundaries."""
    G = hpw.Line(dim)
    qw = hpw.Coined(G, coin=coin_type)
    
    coin = qw.get_coin().toarray()
    shift = qw.get_shift().toarray()
    adj = adjacency_matrix(qw)

    center = dim // 2
    initial_state = qw.state([
        (1, (center, center + 1)), 
        (1j, (center, center - 1))
    ])
    initial_state_matrix = np.outer(initial_state, initial_state.conj())

    return {
        'graph': G, 'qw': qw,
        'initial_state': initial_state,
        'initial_state_matrix': initial_state_matrix,
        'coin': coin, 'shift': shift, 'adjacency': adj,
        'num_vertices': dim,
        'dist_sq': _dist_sq_line(dim, center),
    }


def create_cycle_graph(dim, coin_type='H'):
    """Construct a cycle graph."""
    G = hpw.Cycle(dim)
    qw = hpw.Coined(G, coin=coin_type)
    
    coin = qw.get_coin().toarray()
    shift = qw.get_shift().toarray()
    adj = adjacency_matrix(qw)

    center = dim // 2
    initial_state = qw.state([
        (1, (center, center + 1)), 
        (1j, (center, center - 1))
    ])
    initial_state_matrix = np.outer(initial_state, initial_state.conj())

    return {
        'graph': G, 'qw': qw,
        'initial_state': initial_state,
        'initial_state_matrix': initial_state_matrix,
        'coin': coin, 'shift': shift, 'adjacency': adj,
        'num_vertices': dim,
        'dist_sq': _dist_sq_cycle(dim, center),
    }


def create_grid_graph(dim, coin_type='G'):
    """Construct a periodic 2D grid with diagonal edges."""
    G = hpw.Grid(dim, diagonal=True, periodic=True)
    qw = hpw.Coined(G, coin=coin_type)

    coin = qw.get_coin().toarray()
    shift = qw.get_shift().toarray()
    adj = adjacency_matrix(qw)

    center = np.array((dim // 2, dim // 2))
    initial_state = qw.state([
        (0.5, (center, center + (1, 1))),
        (0.5j, (center, center + (1, -1))),
        (0.5j, (center, center + (-1, 1))),
        (0.5, (center, center + (-1, -1)))
    ])
    initial_state_matrix = np.outer(initial_state, initial_state.conj())

    num_vertices = dim * dim
    dist_sq = np.zeros(num_vertices, dtype=np.float64)
    cx, cy = dim // 2, dim // 2
    for v in range(num_vertices):
        x, y = v // dim, v % dim
        dx = abs(x - cx)
        dy = abs(y - cy)
        dx = min(dx, dim - dx)
        dy = min(dy, dim - dy)
        dist_sq[v] = dx ** 2 + dy ** 2

    return {
        'graph': G, 'qw': qw,
        'initial_state': initial_state,
        'initial_state_matrix': initial_state_matrix,
        'coin': coin, 'shift': shift, 'adjacency': adj,
        'num_vertices': num_vertices,
        'dist_sq': dist_sq,
    }


def create_square_lattice(dim, coin_type='G'):
    """Construct a periodic square lattice of degree 4."""
    nx_graph = nx.grid_2d_graph(dim, dim, periodic=True)
    mapping = {node: i for i, node in enumerate(nx_graph.nodes())}
    nx_graph = nx.relabel_nodes(nx_graph, mapping)
    
    G = hpw.Graph(nx_graph)
    qw = hpw.Coined(G, coin=coin_type)
    
    coin = qw.get_coin().toarray()
    shift = qw.get_shift().toarray()
    adj = adjacency_matrix(qw)

    center_coord = (dim // 2, dim // 2)
    center = mapping[center_coord]
    
    neighbors = []
    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
        neighbor_coord = ((center_coord[0] + dx) % dim, (center_coord[1] + dy) % dim)
        neighbors.append(mapping[neighbor_coord])
    
    amp = 1.0 / 2.0
    state_list = [(amp * (1 if i < 2 else 1j), (center, nb)) for i, nb in enumerate(neighbors)]
    initial_state = qw.state(state_list)
    initial_state_matrix = np.outer(initial_state, initial_state.conj())

    return {
        'graph': G, 'qw': qw,
        'initial_state': initial_state,
        'initial_state_matrix': initial_state_matrix,
        'coin': coin, 'shift': shift, 'adjacency': adj,
        'num_vertices': dim * dim,
        'dist_sq': _dist_sq_periodic_2d(dim * dim, dim, center_coord, mapping),
        'dim': dim,
    }


def create_triangular_lattice(dim, coin_type='G'):
    """Construct a periodic triangular lattice of degree 6."""
    nx_graph = nx.triangular_lattice_graph(dim, dim, periodic=True)
    mapping = {node: i for i, node in enumerate(nx_graph.nodes())}
    nx_graph_relabeled = nx.relabel_nodes(nx_graph, mapping)
    
    G = hpw.Graph(nx_graph_relabeled)
    qw = hpw.Coined(G, coin=coin_type)
    
    coin = qw.get_coin().toarray()
    shift = qw.get_shift().toarray()
    adj = adjacency_matrix(qw)
    
    num_vertices = nx_graph_relabeled.number_of_nodes()
    center_idx = num_vertices // 2
    center_neighbors = list(nx_graph_relabeled.neighbors(center_idx))
    degree = len(center_neighbors)
    
    amp = 1.0 / np.sqrt(degree)
    state_list = [(amp, (center_idx, nb)) for nb in center_neighbors]
    initial_state = qw.state(state_list)
    initial_state_matrix = np.outer(initial_state, initial_state.conj())

    return {
        'graph': G, 'qw': qw,
        'initial_state': initial_state,
        'initial_state_matrix': initial_state_matrix,
        'coin': coin, 'shift': shift, 'adjacency': adj,
        'num_vertices': num_vertices,
        'dist_sq': _dist_sq_triangular_euclidean(mapping, center_idx, dim),
        'dim': dim,
    }


def create_honeycomb_lattice(dim, coin_type='G'):
    """Construct a periodic honeycomb lattice of degree 3."""
    m = dim
    n_col = dim + 1 if dim % 2 == 1 else dim  # Ensure that n_col is even
    nx_graph = nx.hexagonal_lattice_graph(m, n_col, periodic=True)
    mapping = {node: i for i, node in enumerate(nx_graph.nodes())}
    nx_graph_relabeled = nx.relabel_nodes(nx_graph, mapping)
    
    G = hpw.Graph(nx_graph_relabeled)
    qw = hpw.Coined(G, coin=coin_type)
    
    coin = qw.get_coin().toarray()
    shift = qw.get_shift().toarray()
    adj = adjacency_matrix(qw)
    
    num_vertices = nx_graph_relabeled.number_of_nodes()
    center_idx = num_vertices // 2
    center_neighbors = list(nx_graph_relabeled.neighbors(center_idx))
    degree = len(center_neighbors)
    
    amp = 1.0 / np.sqrt(degree)
    state_list = [(amp, (center_idx, nb)) for nb in center_neighbors]
    initial_state = qw.state(state_list)
    initial_state_matrix = np.outer(initial_state, initial_state.conj())

    return {
        'graph': G, 'qw': qw,
        'initial_state': initial_state,
        'initial_state_matrix': initial_state_matrix,
        'coin': coin, 'shift': shift, 'adjacency': adj,
        'num_vertices': num_vertices,
        'dist_sq': _dist_sq_honeycomb_euclidean(mapping, center_idx, m, n_col),  # dim → m, n_col
        'dim': dim,
    }


def create_grid_3d(dim, coin_type='G'):
    """Construct a periodic three-dimensional grid."""
    nx_graph = nx.grid_graph(dim=[dim, dim, dim], periodic=True)
    mapping = {node: i for i, node in enumerate(nx_graph.nodes())}
    nx_graph = nx.relabel_nodes(nx_graph, mapping)
    
    G = hpw.Graph(nx_graph)
    qw = hpw.Coined(G, coin=coin_type)
    
    coin = qw.get_coin().toarray()
    shift = qw.get_shift().toarray()
    adj = adjacency_matrix(qw)

    center_coord = (dim // 2, dim // 2, dim // 2)
    center = mapping[center_coord]
    
    neighbors = []
    for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
        neighbor_coord = (
            (center_coord[0] + dx) % dim,
            (center_coord[1] + dy) % dim,
            (center_coord[2] + dz) % dim
        )
        neighbors.append(mapping[neighbor_coord])
    
    amp = 1.0 / np.sqrt(6)
    state_list = [(amp * (1 if i < 3 else 1j), (center, nb)) for i, nb in enumerate(neighbors)]
    initial_state = qw.state(state_list)
    initial_state_matrix = np.outer(initial_state, initial_state.conj())

    return {
        'graph': G, 'qw': qw,
        'initial_state': initial_state,
        'initial_state_matrix': initial_state_matrix,
        'coin': coin, 'shift': shift, 'adjacency': adj,
        'num_vertices': dim ** 3,
        'dist_sq': _dist_sq_periodic_3d(dim ** 3, dim, center_coord, mapping),
        'dim': dim,
    }


def create_sc_lattice(dim, coin_type='G'):
    """Alias for a periodic simple-cubic lattice of degree 6."""
    return create_grid_3d(dim, coin_type)


def create_bethe_lattice(coordination, depth, coin_type='G'):
    """Bethe lattice (Cayley tree)"""
    nx_graph = nx.balanced_tree(r=coordination - 1, h=depth)
    
    G = hpw.Graph(nx_graph)
    qw = hpw.Coined(G, coin=coin_type)
    
    coin = qw.get_coin().toarray()
    shift = qw.get_shift().toarray()
    adj = adjacency_matrix(qw)
    
    num_vertices = nx_graph.number_of_nodes()
    root = 0
    root_neighbors = list(nx_graph.neighbors(root))
    degree = len(root_neighbors)
    
    amp = 1.0 / np.sqrt(degree)
    state_list = [(amp, (root, nb)) for nb in root_neighbors]
    initial_state = qw.state(state_list)
    initial_state_matrix = np.outer(initial_state, initial_state.conj())

    return {
        'graph': G, 'qw': qw,
        'initial_state': initial_state,
        'initial_state_matrix': initial_state_matrix,
        'coin': coin, 'shift': shift, 'adjacency': adj,
        'num_vertices': num_vertices,
        'dist_sq': _dist_sq_bfs(nx_graph, root),
        'coordination': coordination, 'depth': depth,
    }


def _build_bcc_lattice(dim, periodic=True):
    """Build a three-dimensional BCC lattice."""
    G = nx.Graph()
    node_id = 0
    coord_to_node = {}
    
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                coord_to_node[(i, j, k, 0)] = node_id
                G.add_node(node_id)
                node_id += 1
                coord_to_node[(i, j, k, 1)] = node_id
                G.add_node(node_id)
                node_id += 1
    
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                body_center = coord_to_node[(i, j, k, 1)]
                for di in [0, 1]:
                    for dj in [0, 1]:
                        for dk in [0, 1]:
                            ci = (i + di) % dim if periodic else i + di
                            cj = (j + dj) % dim if periodic else j + dj
                            ck = (k + dk) % dim if periodic else k + dk
                            if periodic or (ci < dim and cj < dim and ck < dim):
                                corner = coord_to_node.get((ci, cj, ck, 0))
                                if corner is not None:
                                    G.add_edge(body_center, corner)
    
    return G, coord_to_node


def create_bcc_lattice(dim, coin_type='G'):
    """Construct a periodic BCC lattice of degree 8."""
    nx_graph, coord_to_node = _build_bcc_lattice(dim, periodic=True)
    
    G = hpw.Graph(nx_graph)
    qw = hpw.Coined(G, coin=coin_type)
    
    coin = qw.get_coin().toarray()
    shift = qw.get_shift().toarray()
    adj = adjacency_matrix(qw)
    
    num_vertices = nx_graph.number_of_nodes()
    center = coord_to_node[(dim // 2, dim // 2, dim // 2, 1)]
    center_neighbors = list(nx_graph.neighbors(center))
    degree = len(center_neighbors)
    
    amp = 1.0 / np.sqrt(degree)
    state_list = [(amp, (center, nb)) for nb in center_neighbors]
    initial_state = qw.state(state_list)
    initial_state_matrix = np.outer(initial_state, initial_state.conj())

    return {
        'graph': G, 'qw': qw,
        'initial_state': initial_state,
        'initial_state_matrix': initial_state_matrix,
        'coin': coin, 'shift': shift, 'adjacency': adj,
        'num_vertices': num_vertices,
        'dist_sq': _dist_sq_bcc(num_vertices, dim, center, coord_to_node),
        'dim': dim,
    }


def _build_fcc_lattice(dim, periodic=True):
    """Build a three-dimensional FCC lattice."""
    G = nx.Graph()
    node_id = 0
    coord_to_node = {}
    
    size = 2 * dim
    for i in range(size):
        for j in range(size):
            for k in range(size):
                if (i + j + k) % 2 == 0:
                    coord_to_node[(i, j, k)] = node_id
                    G.add_node(node_id)
                    node_id += 1
    
    neighbor_offsets = [(1,1,0), (1,-1,0), (-1,1,0), (-1,-1,0),
                        (1,0,1), (1,0,-1), (-1,0,1), (-1,0,-1),
                        (0,1,1), (0,1,-1), (0,-1,1), (0,-1,-1)]
    
    for (i, j, k), node in coord_to_node.items():
        for di, dj, dk in neighbor_offsets:
            ni = (i + di) % size if periodic else i + di
            nj = (j + dj) % size if periodic else j + dj
            nk = (k + dk) % size if periodic else k + dk
            neighbor = coord_to_node.get((ni, nj, nk))
            if neighbor is not None and neighbor > node:
                G.add_edge(node, neighbor)
    
    return G, coord_to_node


def create_fcc_lattice(dim, coin_type='G'):
    """Construct a periodic FCC lattice of degree 12."""
    nx_graph, coord_to_node = _build_fcc_lattice(dim, periodic=True)
    
    G = hpw.Graph(nx_graph)
    qw = hpw.Coined(G, coin=coin_type)
    
    coin = qw.get_coin().toarray()
    shift = qw.get_shift().toarray()
    adj = adjacency_matrix(qw)
    
    num_vertices = nx_graph.number_of_nodes()
    center_coord = (dim, dim, dim)
    center = coord_to_node[center_coord]
    center_neighbors = list(nx_graph.neighbors(center))
    degree = len(center_neighbors)
    
    amp = 1.0 / np.sqrt(degree)
    state_list = [(amp, (center, nb)) for nb in center_neighbors]
    initial_state = qw.state(state_list)
    initial_state_matrix = np.outer(initial_state, initial_state.conj())

    return {
        'graph': G, 'qw': qw,
        'initial_state': initial_state,
        'initial_state_matrix': initial_state_matrix,
        'coin': coin, 'shift': shift, 'adjacency': adj,
        'num_vertices': num_vertices,
        'dist_sq': _dist_sq_fcc(num_vertices, dim, center_coord, coord_to_node),
        'dim': dim,
    }


def _build_uv_flower(u, v, m):
    """Build a (u,v)-flower network."""
    G = nx.MultiGraph()
    G.add_edge(0, 1)
    next_node = 2
    
    for _ in range(m):
        edges_to_replace = list(G.edges(keys=True))
        G.remove_edges_from(edges_to_replace)
        
        for (a, b, key) in edges_to_replace:
            if u == 1:
                G.add_edge(a, b)
            else:
                current = a
                for _ in range(u - 1):
                    G.add_node(next_node)
                    G.add_edge(current, next_node)
                    current = next_node
                    next_node += 1
                G.add_edge(current, b)
            
            if v == 1:
                G.add_edge(a, b)
            else:
                current = a
                for _ in range(v - 1):
                    G.add_node(next_node)
                    G.add_edge(current, next_node)
                    current = next_node
                    next_node += 1
                G.add_edge(current, b)
    
    return G


def create_uv_flower(u, v, generation, coin_type='G'):
    """(u,v)-flower network"""
    multi_graph = _build_uv_flower(u, v, generation)
    nx_graph = nx.Graph(multi_graph)
    
    G = hpw.Graph(nx_graph)
    qw = hpw.Coined(G, coin=coin_type)
    
    coin = qw.get_coin().toarray()
    shift = qw.get_shift().toarray()
    adj = adjacency_matrix(qw)
    
    num_vertices = nx_graph.number_of_nodes()
    root = 0
    root_neighbors = list(nx_graph.neighbors(root))
    degree = len(root_neighbors)
    
    amp = 1.0 / np.sqrt(degree)
    state_list = [(amp, (root, nb)) for nb in root_neighbors]
    initial_state = qw.state(state_list)
    initial_state_matrix = np.outer(initial_state, initial_state.conj())

    return {
        'graph': G, 'qw': qw,
        'initial_state': initial_state,
        'initial_state_matrix': initial_state_matrix,
        'coin': coin, 'shift': shift, 'adjacency': adj,
        'num_vertices': num_vertices,
        'dist_sq': _dist_sq_bfs(nx_graph, root),
        'u': u, 'v': v, 'generation': generation,
    }


def create_graph(graph_type, coin_type='H', **kwargs):
    """
    Dispatch to the constructor for the requested graph type.
    
    Returns:
        dict: graph, qw, initial_state, coin, shift, adjacency, dist_sq, etc.
    """
    if graph_type == 'line':
        return create_line_graph(kwargs['graph_size'], coin_type)
    elif graph_type == 'cycle':
        return create_cycle_graph(kwargs['graph_size'], coin_type)
    elif graph_type == 'grid':
        return create_grid_graph(kwargs['graph_size'], coin_type)
    elif graph_type == 'square':
        return create_square_lattice(kwargs['graph_size'], coin_type)
    elif graph_type == 'triangular':
        return create_triangular_lattice(kwargs['graph_size'], coin_type)
    elif graph_type == 'honeycomb':
        return create_honeycomb_lattice(kwargs['graph_size'], coin_type)
    elif graph_type == 'grid3d':
        return create_grid_3d(kwargs['graph_size'], coin_type)
    elif graph_type == 'sc':
        return create_sc_lattice(kwargs['graph_size'], coin_type)
    elif graph_type == 'bcc':
        return create_bcc_lattice(kwargs['graph_size'], coin_type)
    elif graph_type == 'fcc':
        return create_fcc_lattice(kwargs['graph_size'], coin_type)
    elif graph_type == 'bethe':
        return create_bethe_lattice(kwargs['coordination'], kwargs['depth'], coin_type)
    elif graph_type == 'uv_flower':
        return create_uv_flower(kwargs['u'], kwargs['v'], kwargs['generation'], coin_type)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")


# ============================================================
# Walk simulations with on-the-fly MSD and snapshots
# ============================================================

def random_walk(graph_data, time_length, save_interval=1):
    """
    Simulate a classical random walk.
    Dynamics: M <- S A diag(M)
    """
    A = graph_data['adjacency']
    S = graph_data['shift']
    M = graph_data['initial_state_matrix'].copy()
    qw = graph_data['qw']
    dist_sq = graph_data['dist_sq']
    num_vertices = graph_data['num_vertices']
    pi_uniform = np.ones(num_vertices) / num_vertices

    # Initial probability
    p0 = single_prob(qw, np.diag(M))
    
    msd_list = [np.sum(dist_sq * p0)]
    tvd_list = []
    p_snapshots = [p0.copy()]
    snapshot_times = [0]
    
    p_avg = np.zeros(num_vertices)
    
    for t in range(1, time_length + 1):
        M = np.diag(S @ A @ np.diag(M))
        p_t = single_prob(qw, np.diag(M))
        
        # MSD at every step
        msd_list.append(np.sum(dist_sq * p_t))
        
        # Running time average
        p_avg = (p_avg * (t - 1) + p_t) / t
        
        # TVD at every step
        tvd_list.append(0.5 * np.sum(np.abs(p_avg - pi_uniform)))
        
        # Snapshots at save_interval
        if t % save_interval == 0:
            p_snapshots.append(p_t.copy())
            snapshot_times.append(t)
    
    # Ensure that the final step is included
    if time_length not in snapshot_times:
        p_snapshots.append(p_t.copy())
        snapshot_times.append(time_length)
    
    return {
        'msd_list': np.array(msd_list),
        'tvd_list': np.array(tvd_list),
        'p_snapshots': np.array(p_snapshots),
        'snapshot_times': np.array(snapshot_times),
        'p_final': p_t,
        'p_avg_final': p_avg,
    }


def quantum_walk(graph_data, time_length, save_interval=1):
    """
    Simulate a quantum walk using a density matrix.
    Dynamics: M <- S C M C^dagger S^dagger
    """
    S = graph_data['shift']
    C = graph_data['coin']
    M = graph_data['initial_state_matrix'].copy()
    qw = graph_data['qw']
    dist_sq = graph_data['dist_sq']
    num_vertices = graph_data['num_vertices']
    pi_uniform = np.ones(num_vertices) / num_vertices
    
    ST = S.conj().T
    CT = C.conj().T

    p0 = single_prob(qw, np.diag(M))
    
    msd_list = [np.sum(dist_sq * p0)]
    tvd_list = []
    p_snapshots = [p0.copy()]
    snapshot_times = [0]
    
    p_avg = np.zeros(num_vertices)
    
    for t in range(1, time_length + 1):
        M = S @ C @ M @ CT @ ST
        p_t = single_prob(qw, np.diag(M))
        
        msd_list.append(np.sum(dist_sq * p_t))
        p_avg = (p_avg * (t - 1) + p_t) / t
        tvd_list.append(0.5 * np.sum(np.abs(p_avg - pi_uniform)))
        
        if t % save_interval == 0:
            p_snapshots.append(p_t.copy())
            snapshot_times.append(t)
    
    if time_length not in snapshot_times:
        p_snapshots.append(p_t.copy())
        snapshot_times.append(time_length)
    
    return {
        'msd_list': np.array(msd_list),
        'tvd_list': np.array(tvd_list),
        'p_snapshots': np.array(p_snapshots),
        'snapshot_times': np.array(snapshot_times),
        'p_final': p_t,
        'p_avg_final': p_avg,
    }


def quantum_random_walk_m(graph_data, time_length, n_qw, n_rw, save_interval=1):
    """
    Simulate QRW-M by alternating n_qw QW steps and n_rw RW steps.
    """
    A = graph_data['adjacency']
    S = graph_data['shift']
    C = graph_data['coin']
    M = graph_data['initial_state_matrix'].copy()
    qw = graph_data['qw']
    dist_sq = graph_data['dist_sq']
    num_vertices = graph_data['num_vertices']
    pi_uniform = np.ones(num_vertices) / num_vertices
    
    ST = S.conj().T
    CT = C.conj().T

    p0 = single_prob(qw, np.diag(M))
    
    msd_list = [np.sum(dist_sq * p0)]
    tvd_list = []
    p_snapshots = [p0.copy()]
    snapshot_times = [0]
    
    p_avg = np.zeros(num_vertices)
    
    cycle_length = n_qw + n_rw
    num_cycles = time_length // cycle_length
    t = 0

    for _ in range(num_cycles):
        # Quantum steps
        for _ in range(n_qw):
            t += 1
            M = S @ C @ M @ CT @ ST
            p_t = single_prob(qw, np.diag(M))
            
            msd_list.append(np.sum(dist_sq * p_t))
            p_avg = (p_avg * (t - 1) + p_t) / t
            tvd_list.append(0.5 * np.sum(np.abs(p_avg - pi_uniform)))
            
            if t % save_interval == 0:
                p_snapshots.append(p_t.copy())
                snapshot_times.append(t)
        
        # Random steps
        for _ in range(n_rw):
            t += 1
            M = np.diag(S @ A @ np.diag(M))
            p_t = single_prob(qw, np.diag(M))
            
            msd_list.append(np.sum(dist_sq * p_t))
            p_avg = (p_avg * (t - 1) + p_t) / t
            tvd_list.append(0.5 * np.sum(np.abs(p_avg - pi_uniform)))
            
            if t % save_interval == 0:
                p_snapshots.append(p_t.copy())
                snapshot_times.append(t)
    
    if t not in snapshot_times:
        p_snapshots.append(p_t.copy())
        snapshot_times.append(t)
    
    return {
        'msd_list': np.array(msd_list),
        'tvd_list': np.array(tvd_list),
        'p_snapshots': np.array(p_snapshots),
        'snapshot_times': np.array(snapshot_times),
        'p_final': p_t,
        'p_avg_final': p_avg,
    }


def quantum_random_walk_a(graph_data, time_length, alpha, save_interval=1):
    """
    Simulate QRW-A as a convex combination.
    Dynamics: M <- alpha (QW step) + (1-alpha) (RW step)
    """
    A = graph_data['adjacency']
    S = graph_data['shift']
    C = graph_data['coin']
    M = graph_data['initial_state_matrix'].copy()
    qw = graph_data['qw']
    dist_sq = graph_data['dist_sq']
    num_vertices = graph_data['num_vertices']
    pi_uniform = np.ones(num_vertices) / num_vertices
    
    ST = S.conj().T
    CT = C.conj().T

    p0 = single_prob(qw, np.diag(M))
    
    msd_list = [np.sum(dist_sq * p0)]
    tvd_list = []
    p_snapshots = [p0.copy()]
    snapshot_times = [0]
    
    p_avg = np.zeros(num_vertices)
    
    for t in range(1, time_length + 1):
        qw_step = S @ C @ M @ CT @ ST
        rw_step = np.diag(S @ A @ np.diag(M))
        M = alpha * qw_step + (1 - alpha) * rw_step
        p_t = single_prob(qw, np.diag(M))
        
        msd_list.append(np.sum(dist_sq * p_t))
        p_avg = (p_avg * (t - 1) + p_t) / t
        tvd_list.append(0.5 * np.sum(np.abs(p_avg - pi_uniform)))
        
        if t % save_interval == 0:
            p_snapshots.append(p_t.copy())
            snapshot_times.append(t)
    
    if time_length not in snapshot_times:
        p_snapshots.append(p_t.copy())
        snapshot_times.append(time_length)
    
    return {
        'msd_list': np.array(msd_list),
        'tvd_list': np.array(tvd_list),
        'p_snapshots': np.array(p_snapshots),
        'snapshot_times': np.array(snapshot_times),
        'p_final': p_t,
        'p_avg_final': p_avg,
    }


def run_walk(graph_data, walk_type, time_length, save_interval=1, **kwargs):
    """
    Dispatch to the requested walk simulator.
    
    Parameters:
        walk_type: 'RW', 'QW', 'QRW_M', 'QRW_A'
        save_interval: Probability-distribution snapshot interval
        kwargs: 
            - alpha (for QRW_A)
            - n_qw, n_rw (for QRW_M)
    
    Returns:
        dict: msd_list, tvd_list, p_snapshots, snapshot_times, p_final, p_avg_final
    """
    if walk_type == 'RW':
        return random_walk(graph_data, time_length, save_interval)
    elif walk_type == 'QW':
        return quantum_walk(graph_data, time_length, save_interval)
    elif walk_type == 'QRW_M':
        n_qw = kwargs.get('n_qw', 1)
        n_rw = kwargs.get('n_rw', 1)
        return quantum_random_walk_m(graph_data, time_length, n_qw, n_rw, save_interval)
    elif walk_type == 'QRW_A':
        alpha = kwargs.get('alpha', 0.5)
        return quantum_random_walk_a(graph_data, time_length, alpha, save_interval)
    else:
        raise ValueError(f"Unknown walk_type: {walk_type}")
