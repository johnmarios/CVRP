import json
from pathlib import Path

import osmnx as ox


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_DIRECTORY / "data" / "athens"
OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "outputs" / "athens"

SCENARIO_FILE = Path(__file__).with_name("athens.json")
VEHICLES_FILE = PROJECT_DIRECTORY / "data" / "vehicles.json"
GRAPH_FILE = DATA_DIRECTORY / "athens_network.graphml"
DISTANCE_MATRIX_FILE = DATA_DIRECTORY / "distance_matrix.json"
TRAVEL_TIME_MATRIX_FILE = DATA_DIRECTORY / "travel_time_matrix.json"
SHORTEST_PATHS_FILE = DATA_DIRECTORY / "shortest_paths.json"

CENTER = (37.9815, 23.7335)  # latitude, longitude
RADIUS = 5200  # meters


def load_scenario():
    """Load the Athens scenario."""
    return json.loads(SCENARIO_FILE.read_text(encoding="utf-8"))


def load_vehicle_types():
    """Load the common vehicle parameters."""
    data = json.loads(VEHICLES_FILE.read_text(encoding="utf-8"))
    return data["vehicle_types"]


def load_road_network():
    """Load the saved Athens road network or download it once."""
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if GRAPH_FILE.exists():
        print("Loading saved Athens road network...")
        graph = ox.load_graphml(GRAPH_FILE)
        save_graph = False
    else:
        print("Downloading Athens road network...")
        graph = ox.graph_from_point(CENTER, dist=RADIUS, network_type="drive", simplify=True,)
        save_graph = True

    if not all("travel_time" in data for _, _, data in graph.edges(data=True)):
        print("Adding road speeds and travel times...")
        graph = ox.routing.add_edge_speeds(graph) # km/h
        graph = ox.routing.add_edge_travel_times(graph) # distance/speed
        save_graph = True

    if save_graph: ox.save_graphml(graph, GRAPH_FILE)
    return graph


def find_nearest_road_node(graph, coordinates):
    """Return the nearest road node without optional scikit-learn code."""
    longitude, latitude = coordinates
    return min(
        graph.nodes,
        key=lambda node: ox.distance.great_circle(
            latitude,
            longitude,
            graph.nodes[node]["y"],
            graph.nodes[node]["x"],
        ),
    )


def connect_locations_to_network(graph, locations):
    """Connect every depot or customer to its nearest road node."""
    return {
        location: find_nearest_road_node(graph, data["coordinates"])
        for location, data in locations.items()
    }


def get_shortest_parallel_edge(graph, first_node, second_node):
    """Return the shortest directed edge between two road nodes."""
    edges = graph.get_edge_data(first_node, second_node)
    return min(edges.values(), key=lambda edge: float(edge["length"]))


def calculate_path_distance(graph, path):
    """Return a road path's distance in kilometres."""
    meters = sum(
        float(get_shortest_parallel_edge(graph, first, second)["length"])
        for first, second in zip(path[:-1], path[1:])
    )
    return meters / 1000


def calculate_path_travel_time(graph, path):
    """Return a road path's base travel time in minutes."""
    seconds = sum(
        float(get_shortest_parallel_edge(graph, first, second)["travel_time"])
        for first, second in zip(path[:-1], path[1:])
    )
    return seconds / 60


def build_path_data(graph, location_nodes):
    """Calculate directed shortest paths, distances, and travel times."""
    distances = {}
    travel_times = {}
    shortest_paths = {}

    for origin, origin_node in location_nodes.items():
        print(f"Calculating paths from {origin}...")
        distances[origin] = {}
        travel_times[origin] = {}
        shortest_paths[origin] = {}

        for destination, destination_node in location_nodes.items():
            if origin == destination:
                path = [origin_node]
            else:
                path = ox.routing.shortest_path(
                    graph,
                    origin_node,
                    destination_node,
                    weight="length",
                )
                if path is None:
                    raise RuntimeError(f"No road path from {origin} to {destination}.")

            distances[origin][destination] = round(
                calculate_path_distance(graph, path),
                6,
            )
            travel_times[origin][destination] = round(
                calculate_path_travel_time(graph, path),
                6,
            )
            shortest_paths[origin][destination] = path

    return distances, travel_times, shortest_paths


def get_path_data(graph, location_nodes):
    """Load cached path data or calculate and save it once."""
    files = (DISTANCE_MATRIX_FILE, TRAVEL_TIME_MATRIX_FILE, SHORTEST_PATHS_FILE,)

    if all(path.exists() for path in files):
        print("Loading saved shortest-path data...")
        distance_matrix = json.loads(DISTANCE_MATRIX_FILE.read_text(encoding="utf-8"))
        travel_time_matrix = json.loads(TRAVEL_TIME_MATRIX_FILE.read_text(encoding="utf-8"))
        shortest_paths = json.loads(SHORTEST_PATHS_FILE.read_text(encoding="utf-8"))
        return distance_matrix, travel_time_matrix, shortest_paths

    distance_matrix, travel_time_matrix, shortest_paths = build_path_data(graph, location_nodes,)
    DISTANCE_MATRIX_FILE.write_text(json.dumps(distance_matrix, indent=2), encoding="utf-8",)
    TRAVEL_TIME_MATRIX_FILE.write_text(json.dumps(travel_time_matrix, indent=2), encoding="utf-8",)
    SHORTEST_PATHS_FILE.write_text(json.dumps(shortest_paths, indent=2), encoding="utf-8",)

    return distance_matrix, travel_time_matrix, shortest_paths


def build_road_routes(routes, schedules, shortest_paths, distance_matrix, travel_time_matrix,):
    """Expand every selected MILP arc into its complete road-node path."""
    road_routes = {}

    for vehicle, milp_route in routes.items():
        road_nodes = []
        hops = []

        for schedule_hop in schedules[vehicle]["hops"]:
            origin = schedule_hop["origin"]
            destination = schedule_hop["destination"]
            # shortest paths store all rode nodes between the two MILP nodes, including the origin and destination
            hop_nodes = shortest_paths[origin][destination]
            # exclude the first node of every hop except the first hop, to avoid duplicates
            road_nodes.extend(hop_nodes if not road_nodes else hop_nodes[1:])
            hops.append(
                {
                    **schedule_hop,
                    "distance_km": distance_matrix[origin][destination],
                    "base_travel_time": travel_time_matrix[origin][destination],
                    "road_nodes": hop_nodes,
                }
            )

        # concentrate the information for each vehicle into one dictionary
        road_routes[vehicle] = {
            "milp_route": milp_route,
            "road_nodes": road_nodes,
            "hops": hops,
            "services": schedules[vehicle]["services"],
            "finish_time": schedules[vehicle]["finish_time"],
        }

    return road_routes


def save_road_route_report(road_routes, output_file):
    """Save readable details for the selected real road routes."""
    lines = ["ATHENS ROUTES ON THE ROAD NETWORK", "=================================", ""]

    for vehicle, route_data in road_routes.items():
        hops = route_data["hops"]
        distance = sum(float(hop["distance_km"]) for hop in hops)
        base_time = sum(float(hop["base_travel_time"]) for hop in hops)
        vehicle_time = sum(hop["arrival"] - hop["departure"] for hop in hops)

        lines.extend(
            [
                vehicle,
                "-" * len(vehicle),
                f"MILP route: {' -> '.join(route_data['milp_route'])}",
                f"Road nodes: {len(route_data['road_nodes'])}",
                f"Distance: {distance:.3f} km",
                f"Base travel time: {base_time:.2f} min",
                f"Vehicle travel time: {vehicle_time:.2f} min",
                "hops:",
            ]
        )

        for hop in hops:
            lines.append(
                f"  {hop['origin']:>3} -> {hop['destination']:<3} | "
                f"{hop['distance_km']:6.3f} km | "
                f"{hop['arrival'] - hop['departure']:6.2f} min | "
                f"{len(hop['road_nodes']):3} road nodes"
            )

        service_text = ", ".join(
            f"C{service['customer']} at t={service['service_start']:.1f}"
            for service in route_data["services"]
        )
        lines.extend([f"Service starts: {service_text}", ""])

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines), encoding="utf-8")
    return output_file
