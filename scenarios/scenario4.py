import math
from pathlib import Path
import sys

import pulp


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from model import build_model, get_d_ij
from tools import (
    add_symmetry_breaking,
    calculate_route_costs,
    extract_routes_and_loads,
    find_case,
    format_loads,
    format_routes,
    get_active_capacity,
    get_output_directory,
    get_total_demand,
    make_text_table,
    readjson,
    save_report,
    solve_model,
)
from visualization import plot_solution


DEPOT_CASES = {
    "Central depot": "data/scenario4/central_depot.json",
    "Two local depots": "data/scenario4/two_local_depots.json",
    "Imbalanced depots": "data/scenario4/imbalanced_depots.json",
    "Free depot choice": "data/scenario4/free_depot_choice.json",
}



# input data
def load_inputs():
    """Load the four depot scenarios and the vehicle parameters."""
    scenarios = {
        case_name: readjson(file_name)
        for case_name, file_name in DEPOT_CASES.items()
    }
    vehicle_types = readjson("data/vehicles.json")["vehicle_types"]
    return scenarios, vehicle_types


def available_fleet_text(scenario):
    """Describe vehicle availability separately for every depot."""
    parts = []
    for vehicle_type, information in scenario["vehicle_info"].items():
        for depot, count in information["depots"].items():
            if count > 0:
                parts.append(f"{depot}:{vehicle_type}={count}")
    return ", ".join(parts)



# solve one depot case 
def create_result(case_name, scenario, vehicle_types):
    """Create an empty result containing the common depot-case data."""
    total_demand = get_total_demand(scenario)
    capacity = vehicle_types["van"]["capacity"]
    return {
        "case": case_name,
        "short_name": scenario["short_name"],
        "available_fleet": available_fleet_text(scenario),
        "selected_fleet": None,
        "status": "Not solved",
        "active_vehicles": None,
        "active_capacity": None,
        "capacity_lower_bound": math.ceil(total_demand / capacity),
        "utilization_percent": None,
        "total_distance": None,
        "total_travel_time": None,
        "fixed_cost": None,
        "distance_cost": None,
        "time_cost": None,
        "objective": None,
        "runtime_seconds": 0.0,
        "routes": {},
        "loads": {},
        "depot_summary": {},
    }


def build_depot_summary(scenario, routes, loads):
    """Aggregate selected vehicles, demand and customers by depot."""
    summary = {
        depot: {
            "vehicles": 0,
            "demand": 0,
            "distance": 0.0,
            "customers": [],
        }
        for depot in scenario["depots"]
    }

    for vehicle, route in routes.items():
        depot = vehicle.split("_")[1]
        customers = [node for node in route if node in scenario["customers"]]
        distance = sum(
            get_d_ij(scenario, first, second)
            for first, second in zip(route[:-1], route[1:])
        )
        summary[depot]["vehicles"] += 1
        summary[depot]["demand"] += loads[vehicle]
        summary[depot]["distance"] += distance
        summary[depot]["customers"].extend(customers)

    return summary


def selected_fleet_text(depot_summary):
    """Describe the number of selected vans at each depot."""
    selected = [
        f"{depot}:van={data['vehicles']}"
        for depot, data in depot_summary.items()
        if data["vehicles"] > 0
    ]
    return ", ".join(selected)


def complete_result(result, model, scenario, vehicle_types, variables):
    """Add routes, depot allocation and costs to one optimal result."""
    routes, loads = extract_routes_and_loads(scenario, variables)
    costs = calculate_route_costs(scenario, vehicle_types, routes)
    depot_summary = build_depot_summary(scenario, routes, loads)
    total_demand = get_total_demand(scenario)
    active_capacity = get_active_capacity(routes, vehicle_types)

    result.update(costs)
    result.update(
        {
            "selected_fleet": selected_fleet_text(depot_summary),
            "active_vehicles": len(routes),
            "active_capacity": active_capacity,
            "utilization_percent": round(100 * total_demand / active_capacity, 2),
            "objective": round(pulp.value(model.objective), 4),
            "routes": routes,
            "loads": loads,
            "depot_summary": depot_summary,
        }
    )


def solve_depot_case(case_name, scenario, vehicle_types):
    """Build, solve and summarize one depot-location case."""
    model, variables = build_model(scenario, vehicle_types)
    add_symmetry_breaking(model, variables, scenario)
    result = create_result(case_name, scenario, vehicle_types)
    result["status"], result["runtime_seconds"] = solve_model(model)

    if result["status"] == "Optimal":
        complete_result(result, model, scenario, vehicle_types, variables)

    return result, variables



# depot experiment
def run_depot_experiment(scenarios, vehicle_types):
    """Solve the four depot-location cases."""
    results = []
    solved_cases = {}

    for case_name, scenario in scenarios.items():
        result, variables = solve_depot_case(
            case_name, scenario, vehicle_types
        )
        results.append(result)
        solved_cases[case_name] = (scenario, variables)
        print(
            f"{case_name}: {result['status']} | "
            f"selected={result['selected_fleet'] or '-'} | "
            f"cost={result['objective'] or '-'}"
        )

    return results, solved_cases


# text report 
def format_depot_summary(summary):
    """Format the depot allocation of one solution."""
    lines = []
    for depot, data in summary.items():
        customers = ",".join(f"C{customer}" for customer in data["customers"])
        lines.append(
            f"{depot}: vehicles={data['vehicles']}, demand={data['demand']}, "
            f"distance={data['distance']:.3f}, customers={customers or '-'}"
        )
    return "\n".join(lines)


def build_report(results):
    """Build the single report for Scenario 4."""
    columns = [
        ("case", "Case"),
        ("available_fleet", "Available fleet"),
        ("selected_fleet", "Selected fleet"),
        ("active_vehicles", "Vehicles"),
        ("total_distance", "Distance"),
        ("fixed_cost", "Fixed cost"),
        ("distance_cost", "Distance cost"),
        ("time_cost", "Time cost"),
        ("objective", "Total cost"),
    ]

    route_lines = []
    for result in results:
        route_lines.extend(
            [
                f"{result['case']}:",
                f"Routes: {format_routes(result['routes'])}",
                f"Loads:  {format_loads(result['loads'])}",
                format_depot_summary(result["depot_summary"]),
                "",
            ]
        )

    central = find_case(results, "Central depot")

    route_text = "\n".join(route_lines).rstrip()
    report_str = (
        "SCENARIO 4 - MULTIPLE DEPOTS AND VEHICLE ALLOCATION\n"
        "====================================================\n"
        "\n"
        "EXPERIMENTAL SETUP\n"
        "------------------\n"
        "Customers: 6 in two spatial clusters\n"
        "Total demand: 22\n"
        f"Capacity lower bound: {central['capacity_lower_bound']} vans\n"
        "\n"
        "DEPOT COMPARISON\n"
        "----------------\n"
        + make_text_table(results, columns) + "\n"
        "\n"
        "ROUTES AND DEPOT ALLOCATION\n"
        "---------------------------\n"
        f"{route_text}"
    )
    return report_str


# outputs
def save_route_plots(solved_cases, vehicle_types, output_directory, show):
    """Save the route plot of each depot case."""
    route_files = {}
    for case_name, file_name in (
        ("Central depot", "route_central_depot.png"),
        ("Two local depots", "route_two_local_depots.png"),
        ("Imbalanced depots", "route_imbalanced_depots.png"),
        ("Free depot choice", "route_free_depot_choice.png"),
    ):
        scenario, variables = solved_cases[case_name]
        route_file = output_directory / file_name
        plot_solution(
            scenario,
            variables,
            vehicle_types,
            route_file,
            show=show,
        )
        route_files[case_name] = route_file

    return route_files


# main workflow 
def main():
    """Run Scenario 4 and save its presentation-ready outputs."""
    show_plots = "--show" in sys.argv
    scenarios, vehicle_types = load_inputs()
    output_directory = get_output_directory("scenario4")

    print("Running multiple-depot comparison...")
    results, solved_cases = run_depot_experiment(
        scenarios, vehicle_types
    )
    report = build_report(results)
    report_file = save_report(
        report, output_directory, "scenario4_results.txt"
    )
    route_files = save_route_plots(
        solved_cases,
        vehicle_types,
        output_directory,
        show_plots,
    )

    print("\n" + report)
    print(f"Results saved in: {report_file}")
    for case_name, route_file in route_files.items():
        print(f"{case_name} route saved in: {route_file}")


if __name__ == "__main__":
    main()
