import matplotlib.pyplot as plt
import networkx as nx

# Function to draw a network topology
def draw_graph(G, title, pos=None):
    plt.figure(figsize=(8, 8))
    plt.title(title)
    if pos is None:
        pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=500, font_size=10, node_color='lightblue', edge_color='gray')
    plt.show()

# Line Topology
G_line = nx.path_graph(5)
draw_graph(G_line, "Line Topology")

# Ring Topology
G_ring = nx.cycle_graph(5)
draw_graph(G_ring, "Ring Topology")

# Mesh Topology (2D Grid)
G_mesh = nx.grid_2d_graph(3, 3)  # 3x3 Grid
# Convert grid graph to regular graph to draw
G_mesh = nx.convert_node_labels_to_integers(G_mesh)
draw_graph(G_mesh, "Mesh Topology")

# Torus Topology
G_torus = nx.grid_2d_graph(3, 3, periodic=True)  # 3x3 Grid with periodic boundary conditions
draw_graph(G_torus, "Torus Topology")

# Cube (3D Grid) Topology
G_cube = nx.grid_graph(dim=[2, 2, 2])  # 2x2x2 Grid, which is a hypercube of dimension 3
G_cube = nx.convert_node_labels_to_integers(G_cube)  # Convert to a graph with integer labels
draw_graph(G_cube, "Cube (3D Grid) Topology")

# Define CCC function used in the above code
def create_ccc_graph(dimensions):
    # Your CCC function here
    pass


# Cube-Connected Cycles (CCC)
# G_ccc = create_ccc_graph(3)
# draw_graph(G_ccc, "Cube-Connected Cycles (CCC)")

# Cube-Connected Cycles (CCC)
G_ccc = create_ccc_graph(3)
print("Nodes:", G_ccc.nodes())
print("Edges:", G_ccc.edges())

# This will attempt to create a position map before drawing.
# If it fails, it will let you know that the problem is with the spring_layout
try:
    pos = nx.spring_layout(G_ccc)
except Exception as e:
    print("An error occurred with spring_layout:", e)
    pos = None  # Fall back to None so you can still attempt to draw the graph

draw_graph(G_ccc, "Cube-Connected Cycles (CCC)", pos=pos)

# Hypercube Topology
G_hypercube = nx.hypercube_graph(4)  # 4-dimensional hypercube
draw_graph(G_hypercube, "Hypercube Topology")

# Star Topology
G_star = nx.star_graph(5)
draw_graph(G_star, "Star Topology")

# Tree Topology
G_tree = nx.balanced_tree(2, 3)
draw_graph(G_tree, "Tree Topology")

# Fully Connected Network (Complete Graph)
G_complete = nx.complete_graph(5)
draw_graph(G_complete, "Fully Connected Network")

# Bipartite Graph
G_bipartite = nx.complete_bipartite_graph(3, 4)
draw_graph(G_bipartite, "Bipartite Graph")

# Wheel Graph
G_wheel = nx.wheel_graph(6)
draw_graph(G_wheel, "Wheel Graph")

# Barbell Graph
G_barbell = nx.barbell_graph(5, 1)
draw_graph(G_barbell, "Barbell Graph")

# Ladder Graph
G_ladder = nx.ladder_graph(5)
draw_graph(G_ladder, "Ladder Graph")



# Cube-Connected Cycles (CCC)
def create_ccc_graph(dimensions):
    H = nx.hypercube_graph(dimensions)
    CCC = nx.Graph()
    for node in H.nodes():
        for i in range(dimensions):
            CCC.add_node((node, i))
            CCC.add_edge((node, i), ((node, (i + 1) % dimensions)))
    for edge in H.edges():
        for i in range(dimensions):
            CCC.add_edge((edge[0], i), (edge[1], i))
    return CCC

G_ccc = create_ccc_graph(3)
draw_graph(G_ccc, "Cube-Connected Cycles (CCC)")




# Hypercube Topology
G_hypercube = nx.hypercube_graph(4)  # 4-dimensional hypercube
draw_graph(G_hypercube, "Hypercube Topology")

