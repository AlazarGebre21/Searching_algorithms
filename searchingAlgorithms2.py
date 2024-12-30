import sys
import networkx as nx
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QHBoxLayout,
    QLineEdit, QLabel, QComboBox, QSplitter, QListWidget, QMessageBox, QInputDialog
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import ast  # For parsing dictionary input
import heapq  # For UCS, A*, and Greedy

class GraphCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_graph(self, graph, path=None, visited_nodes=None, visited_edges=None, node_color='skyblue', edge_color='gray', heuristics=None, path_nodes_color='red'):
        self.ax.clear()
        pos = nx.spring_layout(graph, k=1.5)  # Spreads nodes further apart
        edge_labels = nx.get_edge_attributes(graph, 'weight')  # To display weights
        node_labels = nx.get_node_attributes(graph, 'heuristic')  # For node heuristics
        
        # Draw all nodes and edges in their default color
        nx.draw(graph, pos, ax=self.ax, with_labels=True, node_color=node_color, edge_color=edge_color, node_size=700, font_size=10)
        
        if path:
            # Highlight the nodes in the path with a different color
            path_nodes = set(path)  # Set of nodes that are part of the path
            nx.draw_networkx_nodes(graph, pos, nodelist=path_nodes, ax=self.ax, node_color=path_nodes_color, node_size=700)
            
            # Highlight the edges in the path
            edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            nx.draw_networkx_edges(graph, pos, edgelist=edges, ax=self.ax, edge_color='red', width=3)

        if visited_nodes:
            # Highlight visited nodes
            visited_nodes_not_in_path = visited_nodes - path_nodes  # Remove path nodes from visited
            nx.draw_networkx_nodes(graph, pos, nodelist=visited_nodes_not_in_path, ax=self.ax, node_color='lightgreen', node_size=700)
        
        if visited_edges:
            # Highlight visited edges
            nx.draw_networkx_edges(graph, pos, edgelist=visited_edges, ax=self.ax, edge_color='orange', width=2)

        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=self.ax)
        nx.draw_networkx_labels(graph, pos, labels=node_labels, font_color='white', ax=self.ax)
        self.draw()

class GraphApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Graph Search Algorithms GUI")
        self.setGeometry(100, 100, 1200, 700)

        self.graph = nx.Graph()
        self.heuristics = {}

        # Main Layout
        main_layout = QHBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Left Panel for Controls
        control_panel = QVBoxLayout()
        main_layout.addLayout(control_panel, 2)

        # Graph Display Area
        self.graph_canvas = GraphCanvas(self)
        main_layout.addWidget(self.graph_canvas, 5)

        # Graph input section
        self.graph_input_field = QLineEdit(self)
        self.graph_input_field.setPlaceholderText("Enter graph as dictionary format")
        control_panel.addWidget(QLabel("Graph Input (Dict Format):"))
        control_panel.addWidget(self.graph_input_field)

        load_graph_btn = QPushButton("Load Graph", self)
        load_graph_btn.clicked.connect(self.load_graph)
        control_panel.addWidget(load_graph_btn)

        # Heuristic input section
        self.heuristic_input_field = QLineEdit(self)
        self.heuristic_input_field.setPlaceholderText("Enter heuristics as dictionary format")
        control_panel.addWidget(QLabel("Heuristic Input (Dict Format):"))
        control_panel.addWidget(self.heuristic_input_field)

        load_heuristics_btn = QPushButton("Load Heuristics", self)
        load_heuristics_btn.clicked.connect(self.load_heuristics)
        control_panel.addWidget(load_heuristics_btn)

        # Node Operations
        self.node_input = QLineEdit(self)
        self.node_input.setPlaceholderText("Enter Node Name")
        control_panel.addWidget(QLabel("Node Operations:"))
        control_panel.addWidget(self.node_input)

        add_node_btn = QPushButton("Add Node", self)
        add_node_btn.clicked.connect(self.add_node)
        control_panel.addWidget(add_node_btn)

        # Edge Operations
        self.edge_input = QLineEdit(self)
        self.edge_input.setPlaceholderText("Enter Edge (node1,node2,weight)")
        control_panel.addWidget(QLabel("Edge Operations:"))
        control_panel.addWidget(self.edge_input)

        add_edge_btn = QPushButton("Add Edge", self)
        add_edge_btn.clicked.connect(self.add_edge)
        control_panel.addWidget(add_edge_btn)

        # Pathfinding Section
        self.start_node_input = QLineEdit(self)
        self.start_node_input.setPlaceholderText("Start Node")
        self.goal_node_input = QLineEdit(self)
        self.goal_node_input.setPlaceholderText("Goal Node")
        control_panel.addWidget(QLabel("Pathfinding:"))
        control_panel.addWidget(self.start_node_input)
        control_panel.addWidget(self.goal_node_input)

        self.algorithm_selector = QComboBox(self)
        self.algorithm_selector.addItems(["BFS", "DFS", "DLS", "IDDFS", "UCS", "Bidirectional", "A*", "Greedy"])
        control_panel.addWidget(self.algorithm_selector)

        find_path_btn = QPushButton("Find Path", self)
        find_path_btn.clicked.connect(self.find_path)
        control_panel.addWidget(find_path_btn)

        visualize_btn = QPushButton("Visualize Graph", self)
        visualize_btn.clicked.connect(self.visualize_graph)
        control_panel.addWidget(visualize_btn)

    def load_graph(self):
        graph_input = self.graph_input_field.text().strip()
        try:
            graph_dict = ast.literal_eval(graph_input)  # Safely evaluate dictionary input
            if isinstance(graph_dict, dict):
                self.graph.clear()  # Clear the current graph
                for node, edges in graph_dict.items():
                    for neighbor, weight in edges:
                        self.graph.add_edge(node, neighbor, weight=weight)
                self.visualize_graph()
                self.graph_input_field.clear()
            else:
                raise ValueError("Input is not a valid dictionary")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Invalid Graph Input: {e}")

    def load_heuristics(self):
        heuristics_input = self.heuristic_input_field.text().strip()
        try:
            heuristics_dict = ast.literal_eval(heuristics_input)  # Safely evaluate heuristic input
            if isinstance(heuristics_dict, dict):
                self.heuristics = heuristics_dict
                self.heuristic_input_field.clear()
                QMessageBox.information(self, "Success", "Heuristics loaded successfully!")
            else:
                raise ValueError("Input is not a valid dictionary")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Invalid Heuristic Input: {e}")

    def add_node(self):
        node_name = self.node_input.text().strip()
        if node_name:
            if node_name in self.graph:
                QMessageBox.warning(self, "Error", f"Node '{node_name}' already exists!")
            else:
                self.graph.add_node(node_name)
                self.visualize_graph()
                self.node_input.clear()

    def add_edge(self):
        edge_info = self.edge_input.text().strip()
        if edge_info:
            parts = edge_info.split(",")
            if len(parts) == 3:
                node1, node2, weight = parts[0].strip(), parts[1].strip(), float(parts[2].strip())
                if node1 in self.graph and node2 in self.graph:
                    self.graph.add_edge(node1, node2, weight=weight)
                    self.visualize_graph()
                    self.edge_input.clear()
                else:
                    QMessageBox.warning(self, "Error", "Both nodes must exist in the graph!")

    def find_path(self):
        if len(self.graph.nodes) < 2:
            QMessageBox.warning(self, "Error", "Graph must have at least two nodes!")
            return

        start_node = self.start_node_input.text().strip()
        goal_node = self.goal_node_input.text().strip()

        if not start_node or start_node not in self.graph:
            QMessageBox.warning(self, "Error", "Invalid Start Node!")
            return

        if not goal_node or goal_node not in self.graph:
            QMessageBox.warning(self, "Error", "Invalid Goal Node!")
            return

        algorithm = self.algorithm_selector.currentText()  # Get the selected algorithm name

        try:
            path = None
            visited_nodes = set()
            visited_edges = set()

            if algorithm == "BFS":
                path, visited_nodes, visited_edges = self.bfs(start_node, goal_node)
            elif algorithm == "DFS":
                path, visited_nodes, visited_edges = self.dfs(start_node, goal_node)
            elif algorithm == "DLS":
                depth, ok = QInputDialog.getInt(self, "Depth Limit", "Enter Depth Limit:", 1, 1, 100)
                if ok:
                    path, visited_nodes, visited_edges = self.dls(start_node, goal_node, depth)
            elif algorithm == "IDDFS":
                path, visited_nodes, visited_edges = self.iddfs(start_node, goal_node)
            elif algorithm == "UCS":
                path, visited_nodes, visited_edges = self.ucs(start_node, goal_node)
            elif algorithm == "Bidirectional":
                path, visited_nodes, visited_edges = self.bidirectional_search(start_node, goal_node)
            elif algorithm == "A*":
                path, visited_nodes, visited_edges = self.a_star(start_node, goal_node)
            elif algorithm == "Greedy":
                path, visited_nodes, visited_edges = self.greedy_search(start_node, goal_node)

            if path:
                QMessageBox.information(self, "Path Found", f"Path: {' -> '.join(path)}")
                self.graph_canvas.plot_graph(self.graph, path, visited_nodes=visited_nodes, visited_edges=visited_edges, node_color='lightgreen', edge_color='orange', heuristics=self.heuristics, path_nodes_color='cyan')
            else:
                QMessageBox.warning(self, "No Path", "No path found between the selected nodes!")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def visualize_graph(self):
        self.graph_canvas.plot_graph(self.graph, heuristics=self.heuristics)

    # Modified BFS to return visited nodes and edges
    def bfs(self, start, goal):
        visited = {start}
        queue = [(start, [start])]
        visited_nodes = set([start])
        visited_edges = set()
        while queue:
            node, path = queue.pop(0)
            if node == goal:
                return path, visited_nodes, visited_edges
            for neighbor in self.graph.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    visited_nodes.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
                    visited_edges.add((node, neighbor))
        return None, visited_nodes, visited_edges

    # Modified DFS to return visited nodes and edges
    def dfs(self, start, goal):
        visited = {start}
        stack = [(start, [start])]
        visited_nodes = set([start])
        visited_edges = set()
        while stack:
            node, path = stack.pop()
            if node == goal:
                return path, visited_nodes, visited_edges
            for neighbor in self.graph.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    visited_nodes.add(neighbor)
                    stack.append((neighbor, path + [neighbor]))
                    visited_edges.add((node, neighbor))
        return None, visited_nodes, visited_edges

    # Implementing other search algorithms similarly, making sure to return visited nodes and edges...
    def dls(self, start, goal, depth_limit):
        def dfs_depth_limited(node, goal, depth, path, visited, visited_nodes, visited_edges):
            if depth > depth_limit:
                return None
            visited.add(node)
            visited_nodes.add(node)
            if node == goal:
                return path
            for neighbor in self.graph.neighbors(node):
                if neighbor not in visited:
                    visited_edges.add((node, neighbor))
                    result = dfs_depth_limited(neighbor, goal, depth + 1, path + [neighbor], visited, visited_nodes, visited_edges)
                    if result:
                        return result
            return None

        visited = set()
        visited_nodes = set()
        visited_edges = set()
        path = dfs_depth_limited(start, goal, 0, [start], visited, visited_nodes, visited_edges)
        return path, visited_nodes, visited_edges

    def iddfs(self, start, goal):
        def dls_limited(start, goal, depth_limit):
            return self.dls(start, goal, depth_limit)

        visited_nodes = set()
        visited_edges = set()
        for depth in range(1, len(self.graph.nodes) + 1):  # Increase depth incrementally
            path, nodes, edges = dls_limited(start, goal, depth)
            visited_nodes.update(nodes)
            visited_edges.update(edges)
            if path:
                return path, visited_nodes, visited_edges
        return None, visited_nodes, visited_edges

    def ucs(self, start, goal):
        visited = set()
        queue = [(0, start, [start])]  # (cost, node, path)
        visited_nodes = set([start])
        visited_edges = set()
        
        while queue:
            cost, node, path = heapq.heappop(queue)
            if node == goal:
                return path, visited_nodes, visited_edges
            for neighbor in self.graph.neighbors(node):
                edge_weight = self.graph[node][neighbor]['weight']
                if neighbor not in visited:
                    visited.add(neighbor)
                    visited_nodes.add(neighbor)
                    queue.append((cost + edge_weight, neighbor, path + [neighbor]))
                    visited_edges.add((node, neighbor))
                    heapq.heapify(queue)  # Reorder the queue based on cost
        return None, visited_nodes, visited_edges

    def bidirectional_search(self, start, goal):
        if start == goal:
            return [start], {start}, set()  # The path is just the start node itself

        # Initialize the two search frontiers
        visited_from_start = {start}
        visited_from_goal = {goal}
        queue_from_start = [(start, [start])]
        queue_from_goal = [(goal, [goal])]
        visited_edges = set()

        while queue_from_start and queue_from_goal:
            # Forward direction
            node_start, path_from_start = queue_from_start.pop(0)
            for neighbor in self.graph.neighbors(node_start):
                if neighbor not in visited_from_start:
                    visited_from_start.add(neighbor)
                    queue_from_start.append((neighbor, path_from_start + [neighbor]))
                    visited_edges.add((node_start, neighbor))

                    # Check if the neighbor is in the backward search frontier
                    if neighbor in visited_from_goal:
                        # Found intersection, reconstruct the path
                        path_from_goal_reversed = next(path for node, path in queue_from_goal if node == neighbor)[::-1]
                        return path_from_start + path_from_goal_reversed, visited_from_start, visited_edges

            # Backward direction
            node_goal, path_from_goal = queue_from_goal.pop(0)
            for neighbor in self.graph.neighbors(node_goal):
                if neighbor not in visited_from_goal:
                    visited_from_goal.add(neighbor)
                    queue_from_goal.append((neighbor, path_from_goal + [neighbor]))
                    visited_edges.add((node_goal, neighbor))

                    # Check if the neighbor is in the forward search frontier
                    if neighbor in visited_from_start:
                        # Found intersection, reconstruct the path
                        path_from_start_reversed = next(path for node, path in queue_from_start if node == neighbor)[::-1]
                        return path_from_start_reversed + path_from_goal, visited_from_goal, visited_edges

        return None, visited_from_start, visited_edges  # No path found

    def a_star(self, start, goal):
        def heuristic(node):
            return self.heuristics.get(node, 0)

        visited = set()
        open_list = [(0 + heuristic(start), 0, start, [start])]  # (f, g, node, path)
        visited_nodes = set([start])
        visited_edges = set()

        while open_list:
            _, g, node, path = heapq.heappop(open_list)
            if node == goal:
                return path, visited_nodes, visited_edges
            for neighbor in self.graph.neighbors(node):
                edge_weight = self.graph[node][neighbor]['weight']
                f = g + edge_weight + heuristic(neighbor)
                if neighbor not in visited:
                    visited.add(neighbor)
                    visited_nodes.add(neighbor)
                    open_list.append((f, g + edge_weight, neighbor, path + [neighbor]))
                    visited_edges.add((node, neighbor))
                    heapq.heapify(open_list)
        return None, visited_nodes, visited_edges

    def greedy_search(self, start, goal):
        def heuristic(node):
            return self.heuristics.get(node, 0)

        visited = set()
        open_list = [(heuristic(start), start, [start])]  # (f, node, path)
        visited_nodes = set([start])
        visited_edges = set()

        while open_list:
            _, node, path = heapq.heappop(open_list)
            if node == goal:
                return path, visited_nodes, visited_edges
            for neighbor in self.graph.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    visited_nodes.add(neighbor)
                    open_list.append((heuristic(neighbor), neighbor, path + [neighbor]))
                    visited_edges.add((node, neighbor))
                    heapq.heapify(open_list)
        return None, visited_nodes, visited_edges


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GraphApp()
    window.show()
    sys.exit(app.exec_())
