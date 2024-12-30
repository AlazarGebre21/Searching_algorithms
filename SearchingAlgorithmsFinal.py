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


class CustomGraphCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def draw_graph(self, graph, path=None, visited_nodes=None, visited_edges=None,
                  node_color='lightblue', edge_color='black', path_color='lime', visited_color='orange'):
        self.ax.clear()
        pos = nx.spring_layout(graph, k=1.0)
        edge_labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw(graph, pos, ax=self.ax, with_labels=True, node_color=node_color, edge_color=edge_color,
                node_size=700, font_size=10)

        if path:
            path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            nx.draw_networkx_nodes(graph, pos, nodelist=path, ax=self.ax, node_color=path_color, node_size=800)
            nx.draw_networkx_edges(graph, pos, edgelist=path_edges, ax=self.ax, edge_color=path_color, width=3)

        if visited_nodes:
            non_path_nodes = visited_nodes - set(path or [])
            nx.draw_networkx_nodes(graph, pos, nodelist=non_path_nodes, ax=self.ax, node_color=visited_color)

        if visited_edges:
            nx.draw_networkx_edges(graph, pos, edgelist=visited_edges, ax=self.ax, edge_color='gray', style='dotted')

        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=self.ax)
        self.draw()


class GraphSearchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Enhanced Graph Search GUI")
        self.setGeometry(100, 100, 1000, 800)

        self.graph = nx.Graph()
        self.heuristics = {}

        main_layout = QVBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Input and controls section
        controls_layout = QHBoxLayout()
        main_layout.addLayout(controls_layout)

        # Graph input fields
        self.graph_input = QLineEdit(self)
        self.graph_input.setPlaceholderText("Enter graph as {'A': [('B', 2), ('C', 3)]}")
        controls_layout.addWidget(QLabel("Graph Input:"))
        controls_layout.addWidget(self.graph_input)

        graph_load_button = QPushButton("Load Graph")
        graph_load_button.clicked.connect(self.load_graph)
        controls_layout.addWidget(graph_load_button)

        # Heuristic input fields
        self.heuristic_input = QLineEdit(self)
        self.heuristic_input.setPlaceholderText("Enter heuristics as {'A': 1, 'B': 2}")
        controls_layout.addWidget(QLabel("Heuristic Input:"))
        controls_layout.addWidget(self.heuristic_input)

        heuristic_load_button = QPushButton("Load Heuristics")
        heuristic_load_button.clicked.connect(self.load_heuristics)
        controls_layout.addWidget(heuristic_load_button)

        # Pathfinding controls
        pathfinding_layout = QHBoxLayout()
        main_layout.addLayout(pathfinding_layout)

        self.start_node_input = QLineEdit(self)
        self.start_node_input.setPlaceholderText("Start Node")
        pathfinding_layout.addWidget(QLabel("Start Node:"))
        pathfinding_layout.addWidget(self.start_node_input)

        self.goal_node_input = QLineEdit(self)
        self.goal_node_input.setPlaceholderText("Goal Node")
        pathfinding_layout.addWidget(QLabel("Goal Node:"))
        pathfinding_layout.addWidget(self.goal_node_input)

        self.algorithm_selector = QComboBox(self)
        # Added "Bidirectional" to the algorithm options
        self.algorithm_selector.addItems(["BFS", "DFS", "DLS", "IDDFS", "UCS", "Bidirectional", "A*", "Greedy"])
        pathfinding_layout.addWidget(QLabel("Algorithm:"))
        pathfinding_layout.addWidget(self.algorithm_selector)

        find_path_button = QPushButton("Find Path")
        find_path_button.clicked.connect(self.find_path)
        pathfinding_layout.addWidget(find_path_button)

        # Graph visualization
        self.canvas = CustomGraphCanvas(self)
        main_layout.addWidget(self.canvas, stretch=1)

    def load_graph(self):
        try:
            graph_dict = ast.literal_eval(self.graph_input.text())
            self.graph.clear()
            for node, edges in graph_dict.items():
                for neighbor, weight in edges:
                    self.graph.add_edge(node, neighbor, weight=weight)
            self.canvas.draw_graph(self.graph)
            QMessageBox.information(self, "Success", "Graph loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Invalid graph input: {e}")

    def load_heuristics(self):
        try:
            self.heuristics = ast.literal_eval(self.heuristic_input.text())
            QMessageBox.information(self, "Success", "Heuristics loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Invalid heuristics input: {e}")

    def find_path(self):
        start = self.start_node_input.text().strip()
        goal = self.goal_node_input.text().strip()
        algorithm = self.algorithm_selector.currentText()

        if start not in self.graph or goal not in self.graph:
            QMessageBox.warning(self, "Error", "Invalid start or goal node!")
            return

        try:
            path, visited_nodes, visited_edges = self.run_algorithm(start, goal, algorithm)
            if path:
                QMessageBox.information(self, "Path Found", f"Path: {' -> '.join(path)}")
                self.canvas.draw_graph(self.graph, path=path, visited_nodes=visited_nodes, visited_edges=visited_edges)
            else:
                QMessageBox.warning(self, "No Path", "No path found between the nodes.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def run_algorithm(self, start, goal, algorithm):
        if algorithm == "BFS":
            return self.bfs(start, goal)
        elif algorithm == "DFS":
            return self.dfs(start, goal)
        elif algorithm == "UCS":
            return self.ucs(start, goal)
        elif algorithm == "DLS":
            depth, ok = QInputDialog.getInt(self, "Depth Limit", "Enter Depth Limit:", 1, 1, 100)
            if ok:
                return self.dls(start, goal, depth)
        elif algorithm == "IDDFS":
                return self.iddfs(start, goal)
        elif algorithm == "A*":
            return self.a_star(start, goal)
        elif algorithm == "Greedy":
            return self.greedy_search(start, goal)
        elif algorithm == "Bidirectional":
            return self.bidirectional_search(start, goal)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def bfs(self, origin, target):
        explored = {origin}
        frontier = [(origin, [origin])]
        explored_nodes = {origin}
        explored_edges = set()

        while frontier:
            current, route = frontier.pop(0)
            if current == target:
                return route, explored_nodes, explored_edges
            for adj in self.graph.neighbors(current):
                if adj not in explored:
                    explored.add(adj)
                    explored_nodes.add(adj)
                    frontier.append((adj, route + [adj]))
                    explored_edges.add((current, adj))

        return None, explored_nodes, explored_edges

    def dfs(self, origin, target):
        explored = {origin}
        stack = [(origin, [origin])]
        explored_nodes = {origin}
        explored_edges = set()

        while stack:
            current, route = stack.pop()
            if current == target:
                return route, explored_nodes, explored_edges
            for adj in self.graph.neighbors(current):
                if adj not in explored:
                    explored.add(adj)
                    explored_nodes.add(adj)
                    stack.append((adj, route + [adj]))
                    explored_edges.add((current, adj))

        return None, explored_nodes, explored_edges

    def ucs(self, origin, target):
        explored = set()
        frontier = [(0, origin, [origin])]  # (cost, node, path)
        heapq.heapify(frontier)
        explored_nodes = set([origin])
        explored_edges = set()

        while frontier:
            cost, current, route = heapq.heappop(frontier)
            if current == target:
                return route, explored_nodes, explored_edges
            if current in explored:
                continue
            explored.add(current)
            for adj in self.graph.neighbors(current):
                edge_cost = self.graph[current][adj]['weight']
                total_cost = cost + edge_cost
                if adj not in explored:
                    heapq.heappush(frontier, (total_cost, adj, route + [adj]))
                    explored_nodes.add(adj)
                    explored_edges.add((current, adj))

        return None, explored_nodes, explored_edges

    def dls(self, origin, target, depth_cap):
        def dfs_limited_depth(current, target, depth, route, explored, explored_nodes, explored_edges):
            if depth > depth_cap:
                return None
            explored.add(current)
            explored_nodes.add(current)
            if current == target:
                return route
            for adj in self.graph.neighbors(current):
                if adj not in explored:
                    explored_edges.add((current, adj))
                    result = dfs_limited_depth(adj, target, depth + 1, route + [adj], explored, explored_nodes, explored_edges)
                    if result:
                        return result
            return None

        explored = set()
        explored_nodes = set()
        explored_edges = set()
        route = dfs_limited_depth(origin, target, 0, [origin], explored, explored_nodes, explored_edges)
        return route, explored_nodes, explored_edges

    def iddfs(self, origin, target):
        def dls_with_limit(origin, target, depth_cap):
            return self.dls(origin, target, depth_cap)

        explored_nodes = set()
        explored_edges = set()
        for depth in range(1, len(self.graph.nodes) + 1):
            route, nodes, edges = dls_with_limit(origin, target, depth)
            explored_nodes.update(nodes)
            explored_edges.update(edges)
            if route:
                return route, explored_nodes, explored_edges
        return None, explored_nodes, explored_edges

    def bidirectional_search(self, origin, target):
        if origin == target:
            return [origin], {origin}, set()

        start_frontier = {origin}
        goal_frontier = {target}
        start_queue = [(origin, [origin])]
        goal_queue = [(target, [target])]
        explored_edges = set()

        while start_queue and goal_queue:
            current_start, route_start = start_queue.pop(0)
            for adj in self.graph.neighbors(current_start):
                if adj not in start_frontier:
                    start_frontier.add(adj)
                    start_queue.append((adj, route_start + [adj]))
                    explored_edges.add((current_start, adj))

                    if adj in goal_frontier:
                        route_goal = next((path for node, path in goal_queue if node == adj), [adj])[::-1]
                        full_route = route_start + route_goal[1:]
                        return full_route, start_frontier, explored_edges

            current_goal, route_goal = goal_queue.pop(0)
            for adj in self.graph.neighbors(current_goal):
                if adj not in goal_frontier:
                    goal_frontier.add(adj)
                    goal_queue.append((adj, route_goal + [adj]))
                    explored_edges.add((current_goal, adj))

                    if adj in start_frontier:
                        route_start = next((path for node, path in start_queue if node == adj), [adj])[::-1]
                        full_route = route_start + route_goal
                        return full_route, start_frontier, explored_edges

        return None, start_frontier, explored_edges

    def a_star(self, origin, target):
        def estimate_cost(node):
            return self.heuristics.get(node, 0)

        explored = set()
        open_set = [(estimate_cost(origin), 0, origin, [origin])]
        heapq.heapify(open_set)
        explored_nodes = set([origin])
        explored_edges = set()

        while open_set:
            f_cost, g_cost, current, route = heapq.heappop(open_set)
            if current == target:
                return route, explored_nodes, explored_edges
            if current in explored:
                continue
            explored.add(current)
            for adj in self.graph.neighbors(current):
                edge_cost = self.graph[current][adj]['weight']
                g_new = g_cost + edge_cost
                f_new = g_new + estimate_cost(adj)
                if adj not in explored:
                    heapq.heappush(open_set, (f_new, g_new, adj, route + [adj]))
                    explored_nodes.add(adj)
                    explored_edges.add((current, adj))

        return None, explored_nodes, explored_edges

    def greedy_search(self, origin, target):
        def estimate_cost(node):
            return self.heuristics.get(node, 0)

        explored = set()
        open_set = [(estimate_cost(origin), origin, [origin])]
        heapq.heapify(open_set)
        explored_nodes = set([origin])
        explored_edges = set()

        while open_set:
            f_cost, current, route = heapq.heappop(open_set)
            if current == target:
                return route, explored_nodes, explored_edges
            if current in explored:
                continue
            explored.add(current)
            for adj in self.graph.neighbors(current):
                if adj not in explored:
                    heapq.heappush(open_set, (estimate_cost(adj), adj, route + [adj]))
                    explored_nodes.add(adj)
                    explored_edges.add((current, adj))

        return None, explored_nodes, explored_edges


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GraphSearchApp()
    window.show()
    sys.exit(app.exec_())
