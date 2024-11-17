from collections import deque

def bfs(graph, start, goal):
    """
    Perform Breadth-First Search (BFS) on a graph.

    Args:
        graph (dict): Adjacency list representing the graph.
        start (str): The starting node.
        goal (str): The target node.

    Returns:
        list: The path from start to goal, or an empty list if no path exists.
    """
    # Create a queue for BFS
    queue = deque([[start]])
    # Keep track of visited nodes
    visited = set()

    while queue:
        # Get the first path from the queue
        path = queue.popleft()
        # Get the last node from the path
        node = path[-1]

        # Check if the node is the goal
        if node == goal:
            return path

        # If node has not been visited, explore its neighbors
        if node not in visited:
            visited.add(node)
            for neighbor in graph.get(node, []):
                new_path = path + [neighbor]
                queue.append(new_path)

    # Return an empty list if no path exists
    return []

def menu():
    """
    Display the menu and handle user input for searching algorithms.
    """
    graph = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }

    while True:
        print("\n--- Searching Algorithms ---")
        print("1. Breadth-First Search (BFS)")
        print("2. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            start = input("Enter the start node: ")
            goal = input("Enter the goal node: ")

            if start not in graph or goal not in graph:
                print("Invalid nodes! Please try again.")
            else:
                path = bfs(graph, start, goal)
                if path:
                    print(f"Path from {start} to {goal}: {' -> '.join(path)}")
                else:
                    print(f"No path exists from {start} to {goal}.")
        elif choice == '2':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice! Please select again.")

if __name__ == "__main__":
    menu()