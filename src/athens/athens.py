from pathlib import Path
import sys


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from visualization import plot_solution

from athens.animation import save_route_animation
from athens.helpers import (
    OUTPUT_DIRECTORY,
    build_road_routes,
    connect_locations_to_network,
    get_path_data,
    load_road_network,
    load_scenario,
    load_vehicle_types,
    save_road_route_report,
)
from athens.solver import solve_athens_model
from athens.visualization import (
    LABEL_POSITIONS,
    build_solver_graph,
    plot_road_network,
    plot_road_routes,
    plot_solver_graph,
)


def main():
    """Run the complete Athens scenario from network to animations."""
    scenario = load_scenario()
    vehicle_types = load_vehicle_types()
    road_graph = load_road_network()

    depot_nodes = connect_locations_to_network(road_graph, scenario["depots"])
    customer_nodes = connect_locations_to_network(road_graph, scenario["customers"])
    location_nodes = {**depot_nodes, **customer_nodes}

    plot_road_network(
        road_graph,
        scenario,
        depot_nodes,
        customer_nodes,
        OUTPUT_DIRECTORY / "01_road_network.png",
    )

    distance_matrix, travel_time_matrix, shortest_paths = get_path_data(road_graph, location_nodes,)
    scenario["distance_matrix"] = distance_matrix
    scenario["travel_time_matrix"] = travel_time_matrix

    milp_graph = build_solver_graph(scenario, distance_matrix, travel_time_matrix,)
    plot_solver_graph(milp_graph, OUTPUT_DIRECTORY / "02_milp_graph.png",)

    variables, routes, schedules, result = solve_athens_model(scenario, vehicle_types, OUTPUT_DIRECTORY / "03_milp_solution.txt",)

    if result is None: return

    # keep the time the customer is served
    service_times = {service["customer"]: service["service_start"] for schedule in schedules.values() for service in schedule["services"]}

    # keep the total load of each vehicle
    loads = {vehicle: sum(scenario["customers"][customer]["demand"] for customer in route[1:-1]) for vehicle, route in routes.items()}

    # plot solved milp 
    plot_solution(
        scenario,
        variables,
        vehicle_types,
        OUTPUT_DIRECTORY / "03_milp_solution.png",
        show=False,
        show_time_windows=True,
        service_times=service_times,
        label_positions=LABEL_POSITIONS,
        label_font_size=7.5,
        figure_size=(11, 8.5),
    )

    # find real road routes for each vehicle 
    road_routes = build_road_routes(
        routes,
        schedules,
        shortest_paths,
        distance_matrix,
        travel_time_matrix,
    )
    save_road_route_report(road_routes, OUTPUT_DIRECTORY / "04_road_routes.txt",)
    plot_road_routes(
        road_graph,
        scenario,
        road_routes,
        loads,
        vehicle_types,
        depot_nodes,
        customer_nodes,
        OUTPUT_DIRECTORY / "04_road_routes.png",
    )

    # animations of the routes on the MILP graph and on the real road network
    save_route_animation(
        scenario,
        vehicle_types,
        road_routes,
        loads,
        OUTPUT_DIRECTORY / "05_milp_animation.gif",
    )
    save_route_animation(
        scenario,
        vehicle_types,
        road_routes,
        loads,
        OUTPUT_DIRECTORY / "06_road_network_animation.gif",
        graph=road_graph,
        location_nodes=location_nodes,
    )


if __name__ == "__main__":
    main()
