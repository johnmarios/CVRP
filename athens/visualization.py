from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import osmnx as ox

from .helpers import get_shortest_parallel_edge


LABEL_POSITIONS = {
    "1": (-12, 12, "right", "bottom"),
    "2": (-12, 2, "right", "bottom"),
    "3": (0, -12, "center", "top"),
    "4": (10, 4, "left", "bottom"),
    "5": (8, 5, "left", "bottom"),
    "6": (10, 8, "left", "bottom"),
    "7": (8, 6, "left", "bottom"),
    "8": (8, 6, "left", "bottom"),
    "9": (8, 6, "left", "bottom"),
    "10": (8, 6, "left", "bottom"),
    "11": (8, 6, "left", "bottom"),
    "12": (8, 6, "left", "bottom"),
}


def save_plot(fig, output_file):
    """Save one plot and close its figure."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved in: {output_file}")


def vehicle_colors(vehicles):
    """Assign one stable colour to every active vehicle."""
    color_map = plt.get_cmap("tab10")
    return {vehicle: color_map(index) for index, vehicle in enumerate(vehicles)}


def node_position(graph, node):
    """Return a road node as a longitude-latitude pair."""
    return graph.nodes[node]["x"], graph.nodes[node]["y"]


def edge_coordinates(graph, first_node, second_node):
    """Return the directed geometry of one road edge."""
    edge = get_shortest_parallel_edge(graph, first_node, second_node)
    geometry = edge.get("geometry")

    if geometry is None:
        return [node_position(graph, first_node), node_position(graph, second_node)]

    coordinates = list(geometry.coords)
    start = node_position(graph, first_node)
    distance_to_first = (
        (coordinates[0][0] - start[0]) ** 2
        + (coordinates[0][1] - start[1]) ** 2
    )
    distance_to_last = (
        (coordinates[-1][0] - start[0]) ** 2
        + (coordinates[-1][1] - start[1]) ** 2
    )

    if distance_to_last < distance_to_first:
        coordinates.reverse()
    return coordinates


def route_coordinates(graph, road_nodes):
    """Return all coordinates along a complete road-node path."""
    if len(road_nodes) == 1:
        return [node_position(graph, road_nodes[0])]

    coordinates = []
    for first, second in zip(road_nodes[:-1], road_nodes[1:]):
        edge_points = edge_coordinates(graph, first, second)
        coordinates.extend(edge_points if not coordinates else edge_points[1:])
    return coordinates


def plot_road_network(
    graph,
    scenario,
    depot_nodes,
    customer_nodes,
    output_file,
):
    """Plot the original Athens road network and all service locations."""
    fig, ax = ox.plot_graph(
        graph,
        node_size=0,
        edge_color="lightgray",
        edge_linewidth=0.5,
        bgcolor="white",
        figsize=(11, 9),
        show=False,
        close=False,
    )

    for depot, road_node in depot_nodes.items():
        x, y = node_position(graph, road_node)
        ax.scatter(x, y, marker="s", s=90, color="orange", edgecolors="black", zorder=5)
        ax.annotate(
            f"{depot} - {scenario['depots'][depot]['name']}",
            (x, y),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
            zorder=6,
        )

    for customer, road_node in customer_nodes.items():
        x, y = node_position(graph, road_node)
        ax.scatter(x, y, s=45, color="tab:blue", edgecolors="white", zorder=5)
        ax.annotate(
            f"C{customer}",
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            zorder=6,
        )

    ax.set_title("Athens road network, depots, and customers")
    save_plot(fig, output_file)


def build_solver_graph(scenario, distance_matrix, travel_time_matrix):
    """Create the complete directed graph used by the MILP."""
    graph = nx.DiGraph()

    # add depots
    for depot, data in scenario["depots"].items():
        graph.add_node(depot, kind="depot", coordinates=data["coordinates"])

    # add customers
    for customer, data in scenario["customers"].items():
        graph.add_node(customer, kind="customer", coordinates=data["coordinates"], demand=data["demand"], service_time=data["service_time"],)

    # add edges
    for origin, destinations in distance_matrix.items():
        for destination, distance in destinations.items():
            if origin != destination:
                graph.add_edge(origin, destination, distance=distance, travel_time=travel_time_matrix[origin][destination],)

    return graph


def plot_solver_graph(graph, output_file):
    """Plot the complete directed MILP graph."""
    positions = {node: data["coordinates"] for node, data in graph.nodes(data=True)}

    depots = [node for node, data in graph.nodes(data=True) if data["kind"] == "depot"]
    customers = [node for node, data in graph.nodes(data=True) if data["kind"] == "customer"]

    labels = {node: node if data["kind"] == "depot" else f"C{node}\nq={data['demand']}" for node, data in graph.nodes(data=True)}

    fig, ax = plt.subplots(figsize=(10, 8))

    # draw edges
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        edge_color="gray",
        width=0.6,
        alpha=0.18,
        arrows=True,
        arrowsize=8,
        node_size=100,
        connectionstyle="arc3,rad=0.08",
    )

    # draw depots
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=depots,
        node_shape="s",
        node_size=180,
        node_color="orange",
        edgecolors="black",
        ax=ax,
    )

    # draw customers
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=customers,
        node_size=120,
        node_color="tab:blue",
        edgecolors="white",
        ax=ax,
    )

    # draw labels 
    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=8, ax=ax)

    ax.set_title(f"Directed MILP graph: {graph.number_of_nodes()} nodes and {graph.number_of_edges()} arcs")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.margins(0.10)
    ax.grid(alpha=0.25)
    save_plot(fig, output_file)


def plot_road_routes(
    graph,
    scenario,
    road_routes,
    loads,
    vehicle_types,
    depot_nodes,
    customer_nodes,
    output_file,
):
    """Plot the optimized routes on their real directed road paths."""
    vehicles = sorted(road_routes)
    colors = vehicle_colors(vehicles)
    service_times = {
        service["customer"]: service["service_start"]
        for route_data in road_routes.values()
        for service in route_data["services"]
    }

    fig, ax = ox.plot_graph(
        graph,
        node_size=0,
        edge_color="lightgray",
        edge_linewidth=0.55,
        bgcolor="white",
        figsize=(12, 9),
        show=False,
        close=False,
    )

    all_coordinates = []
    for vehicle in vehicles:
        color = colors[vehicle]
        coordinates = route_coordinates(graph, road_routes[vehicle]["road_nodes"])
        all_coordinates.extend(coordinates)
        ax.plot(
            [point[0] for point in coordinates],
            [point[1] for point in coordinates],
            color=color,
            linewidth=2.6,
            alpha=0.9,
            zorder=3,
        )

        for leg in road_routes[vehicle]["legs"]:
            leg_coordinates = route_coordinates(graph, leg["road_nodes"])
            index = max(1, len(leg_coordinates) // 2)
            ax.annotate(
                "",
                xy=leg_coordinates[index],
                xytext=leg_coordinates[index - 1],
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    linewidth=2.0,
                    mutation_scale=13,
                ),
                zorder=4,
            )

    for depot, road_node in depot_nodes.items():
        x, y = node_position(graph, road_node)
        ax.scatter(x, y, marker="s", s=105, color="orange", edgecolors="black", zorder=6)
        ax.annotate(depot, (x, y), xytext=(7, 7), textcoords="offset points", fontsize=10)

    for customer, road_node in customer_nodes.items():
        x, y = node_position(graph, road_node)
        dx, dy, horizontal, vertical = LABEL_POSITIONS[customer]
        demand = scenario["customers"][customer]["demand"]
        ax.scatter(x, y, s=55, color="white", edgecolors="black", zorder=6)
        ax.annotate(
            f"C{customer}  q={demand}\nt={service_times[customer]:.1f}",
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            ha=horizontal,
            va=vertical,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1),
            zorder=7,
        )

    legend = []
    for vehicle in vehicles:
        vehicle_type = vehicle.split("_")[0]
        capacity = vehicle_types[vehicle_type]["capacity"]
        legend.append(
            Line2D(
                [0],
                [0],
                color=colors[vehicle],
                linewidth=2.6,
                label=f"{vehicle} | load {loads[vehicle]}/{capacity}",
            )
        )
    ax.legend(handles=legend, loc="upper right")
    ax.set_title("Optimized routes on the Athens road network")

    xs = [point[0] for point in all_coordinates]
    ys = [point[1] for point in all_coordinates]
    x_margin = (max(xs) - min(xs)) * 0.08
    y_margin = (max(ys) - min(ys)) * 0.10
    ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
    ax.set_ylim(min(ys) - y_margin, max(ys) + y_margin)
    save_plot(fig, output_file)
