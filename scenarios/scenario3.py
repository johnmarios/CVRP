import math
from pathlib import Path
import sys
import time

import pulp


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from model import build_model, get_d_ij, get_t_ijk
from tools import extract_route, format_loads, format_routes, get_output_directory, make_text_table, readjson
from visualization import plot_solution, plot_time_window_results


TIME_WINDOW_CASES = {
    "No time windows": "data/scenario3/no_windows.json",
    "Wide time windows": "data/scenario3/wide_windows.json",
    "Tight feasible": "data/scenario3/tight_feasible.json",
}


# -----------------------------------------------------------------------------
# 1. INPUT DATA
# -----------------------------------------------------------------------------

def load_inputs():
    """Load the three time-window scenarios and the vehicle parameters."""
    scenarios = {
        case_name: readjson(file_name)
        for case_name, file_name in TIME_WINDOW_CASES.items()
    }
    vehicle_types = readjson("data/vehicles.json")["vehicle_types"]
    return scenarios, vehicle_types


def get_number_of_vans(scenario):
    """Return the number of vans available in the scenario."""
    return sum(scenario["vehicle_info"]["van"]["depots"].values())


# -----------------------------------------------------------------------------
# 2. SOLVE ONE TIME-WINDOW CASE
# -----------------------------------------------------------------------------

def add_symmetry_breaking(model, variables):
    """Activate identical vans in a fixed order."""
    vehicles = sorted(variables["z"])
    for first, second in zip(vehicles[:-1], vehicles[1:]):
        model += variables["z"][first] >= variables["z"][second]


def solve_model(model):
    """Solve a model with CBC and return status and runtime."""
    start = time.perf_counter()
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    runtime = round(time.perf_counter() - start, 4)
    return pulp.LpStatus[model.status], runtime


def create_result(case_name, scenario, vehicle_types):
    """Create an empty result containing the analytical lower bounds."""
    total_demand = sum(
        customer["demand"] for customer in scenario["customers"].values()
    )
    capacity = vehicle_types["van"]["capacity"]
    temporal_lower_bound = scenario["temporal_lower_bound"]
    available_vans = get_number_of_vans(scenario)

    return {
        "case": case_name,
        "short_name": scenario["short_name"],
        "windows": scenario["window_summary"],
        "status": "Not solved",
        "available_vehicles": available_vans,
        "active_vehicles": None,
        "active_capacity": None,
        "capacity_lower_bound": math.ceil(total_demand / capacity),
        "temporal_lower_bound": temporal_lower_bound,
        "utilization_percent": None,
        "total_distance": None,
        "total_travel_time": None,
        "maximum_route_time": None,
        "total_waiting_time": None,
        "fixed_cost": None,
        "distance_cost": None,
        "time_cost": None,
        "objective": None,
        "runtime_seconds": 0.0,
        "routes": {},
        "loads": {},
    }


def build_earliest_schedule(scenario, vehicle, route, vehicle_types):
    """Construct the earliest feasible service schedule for one route."""
    vehicle_type = vehicle.split("_")[0]
    current_time = vehicle_types[vehicle_type]["start_time"]
    current_node = route[0]
    rows = []

    for customer in route[1:-1]:
        arrival = current_time + get_t_ijk(
            scenario, vehicle_types, current_node, customer, vehicle
        )
        window_start, window_end = scenario.get("time_windows", {}).get(
            customer, [0, scenario["horizon"]]
        )
        service_start = max(arrival, window_start)
        rows.append(
            {
                "vehicle": vehicle,
                "customer": customer,
                "window_start": window_start,
                "window_end": window_end,
                "arrival": arrival,
                "service_start": service_start,
                "waiting": service_start - arrival,
            }
        )
        current_time = (
            service_start + scenario["customers"][customer]["service_time"]
        )
        current_node = customer

    return_time = current_time + get_t_ijk(
        scenario, vehicle_types, current_node, route[-1], vehicle
    )
    return rows, return_time


def extract_routes_loads_and_schedules(scenario, vehicle_types, variables):
    """Extract routes, loads and deterministic service schedules."""
    routes = {}
    loads = {}
    schedule_rows = []
    return_times = []

    for vehicle in sorted(variables["z"]):
        if pulp.value(variables["z"][vehicle]) < 0.5:
            continue

        route = extract_route(
            vehicle, variables["x"], len(scenario["customers"])
        )
        load = sum(
            scenario["customers"][customer]["demand"]
            for customer in scenario["customers"]
            if pulp.value(variables["y"][customer, vehicle]) > 0.5
        )
        vehicle_schedule, return_time = build_earliest_schedule(
            scenario, vehicle, route, vehicle_types
        )
        routes[vehicle] = route
        loads[vehicle] = load
        schedule_rows.extend(vehicle_schedule)
        return_times.append(return_time)

    return routes, loads, schedule_rows, return_times


def calculate_route_costs(scenario, vehicle_types, routes):
    """Calculate the objective components of the selected routes."""
    fixed_cost = 0.0
    distance_cost = 0.0
    time_cost = 0.0
    total_distance = 0.0
    total_travel_time = 0.0

    for vehicle, route in routes.items():
        vehicle_type = vehicle.split("_")[0]
        vehicle_data = vehicle_types[vehicle_type]
        fixed_cost += vehicle_data["fixed_cost"]

        for first, second in zip(route[:-1], route[1:]):
            distance = get_d_ij(scenario, first, second)
            travel_time = get_t_ijk(
                scenario, vehicle_types, first, second, vehicle
            )
            total_distance += distance
            total_travel_time += travel_time
            distance_cost += vehicle_data["cost_per_distance"] * distance
            time_cost += vehicle_data["cost_per_time"] * travel_time

    return {
        "fixed_cost": round(fixed_cost, 4),
        "distance_cost": round(distance_cost, 4),
        "time_cost": round(time_cost, 4),
        "total_distance": round(total_distance, 4),
        "total_travel_time": round(total_travel_time, 4),
    }


def complete_result(result, model, scenario, vehicle_types, variables):
    """Add routes, schedules and costs to one optimal result."""
    routes, loads, schedule_rows, return_times = (
        extract_routes_loads_and_schedules(
            scenario, vehicle_types, variables
        )
    )
    costs = calculate_route_costs(scenario, vehicle_types, routes)
    capacity = vehicle_types["van"]["capacity"]
    total_demand = sum(
        customer["demand"] for customer in scenario["customers"].values()
    )

    result.update(costs)
    result.update(
        {
            "active_vehicles": len(routes),
            "active_capacity": len(routes) * capacity,
            "utilization_percent": round(
                100 * total_demand / (len(routes) * capacity), 2
            ),
            "maximum_route_time": round(max(return_times), 2),
            "total_waiting_time": round(
                sum(row["waiting"] for row in schedule_rows), 2
            ),
            "objective": round(pulp.value(model.objective), 4),
            "routes": routes,
            "loads": loads,
        }
    )
    return schedule_rows


def solve_time_window_case(case_name, scenario, vehicle_types):
    """Build, solve and summarize one time-window case."""
    model, variables = build_model(scenario, vehicle_types)
    add_symmetry_breaking(model, variables)
    result = create_result(case_name, scenario, vehicle_types)
    result["status"], result["runtime_seconds"] = solve_model(model)
    schedule_rows = []

    if result["status"] == "Optimal":
        schedule_rows = complete_result(
            result, model, scenario, vehicle_types, variables
        )

    return result, variables, schedule_rows


# -----------------------------------------------------------------------------
# 3. EXPERIMENT
# -----------------------------------------------------------------------------

def run_time_window_experiment(scenarios, vehicle_types):
    """Solve the three time-window cases."""
    results = []
    solved_cases = {}

    for case_name, scenario in scenarios.items():
        result, variables, schedule_rows = solve_time_window_case(
            case_name, scenario, vehicle_types
        )
        results.append(result)
        solved_cases[case_name] = (scenario, variables, schedule_rows)
        print(
            f"{case_name}: {result['status']} | "
            f"vehicles={result['active_vehicles'] or '-'} | "
            f"cost={result['objective'] or '-'}"
        )

    return results, solved_cases


def find_case(results, case_name):
    """Return one result using its case name."""
    return next(result for result in results if result["case"] == case_name)


# -----------------------------------------------------------------------------
# 4. TEXT REPORT
# -----------------------------------------------------------------------------

def format_schedule(schedule_rows):
    """Return service times grouped by vehicle."""
    lines = []
    for vehicle in sorted({row["vehicle"] for row in schedule_rows}):
        visits = [row for row in schedule_rows if row["vehicle"] == vehicle]
        text = ", ".join(
            f"C{row['customer']} at t={row['service_start']:.2f} "
            f"in [{row['window_start']},{row['window_end']}]"
            for row in visits
        )
        lines.append(f"{vehicle}: {text}")
    return "\n".join(lines)


def build_schedule_table(schedule_rows):
    """Create presentation-ready rows for the tight-feasible schedule."""
    return [
        {
            "vehicle": row["vehicle"],
            "customer": f"C{row['customer']}",
            "window": f"[{row['window_start']},{row['window_end']}]",
            "arrival": f"{row['arrival']:.2f}",
            "service": f"{row['service_start']:.2f}",
            "waiting": f"{row['waiting']:.2f}",
        }
        for row in schedule_rows
    ]


def build_report(results, solved_cases):
    """Build the single report for Scenario 3."""
    columns = [
        ("case", "Case"),
        ("windows", "Time windows"),
        ("status", "Status"),
        ("active_vehicles", "Vehicles"),
        ("capacity_lower_bound", "Capacity LB"),
        ("temporal_lower_bound", "Temporal LB"),
        ("total_distance", "Distance"),
        ("total_waiting_time", "Waiting"),
        ("objective", "Total cost"),
    ]
    schedule_columns = [
        ("vehicle", "Vehicle"),
        ("customer", "Customer"),
        ("window", "Window"),
        ("arrival", "Arrival"),
        ("service", "Service t"),
        ("waiting", "Waiting"),
    ]
    no_windows = find_case(results, "No time windows")
    tight_schedule = solved_cases["Tight feasible"][2]

    route_lines = []
    for result in results:
        if result["status"] != "Optimal":
            continue
        schedule = solved_cases[result["case"]][2]
        route_lines.extend(
            [
                f"{result['case']}:",
                f"Routes: {format_routes(result['routes'])}",
                f"Loads:  {format_loads(result['loads'])}",
                format_schedule(schedule),
                "",
            ]
        )

    route_text = "\n".join(route_lines).rstrip()
    schedule_table = make_text_table(
        build_schedule_table(tight_schedule), schedule_columns
    )
    report_str = (
        "SCENARIO 3 - VEHICLE ROUTING WITH TIME WINDOWS\n"
        "================================================\n"
        "\n"
        "EXPERIMENTAL SETUP\n"
        "------------------\n"
        "Customers: 6\n"
        "Available fleet: 3 identical vans\n"
        f"Capacity lower bound: {no_windows['capacity_lower_bound']}\n"
        "\n"
        "TIME-WINDOW COMPARISON\n"
        "----------------------\n"
        + make_text_table(results, columns) + "\n"
        "\n"
        "ROUTES AND SERVICE TIMES\n"
        "------------------------\n"
        f"{route_text}\n"
        "\n"
        "TIGHT-FEASIBLE SCHEDULE\n"
        "-----------------------\n"
        f"{schedule_table}"
    )
    return report_str


# -----------------------------------------------------------------------------
# 5. OUTPUTS
# -----------------------------------------------------------------------------

def save_report(report, output_directory):
    """Save and return the scenario report path."""
    report_file = output_directory / "scenario3_results.txt"
    report_file.write_text(report, encoding="utf-8")
    return report_file


def save_summary_plots(results, solved_cases, vehicle_types, output_directory, show):
    """Save the comparison and route figures."""
    comparison_file = output_directory / "time_window_comparison.png"
    plot_time_window_results(results, comparison_file, show=show)

    route_files = {}
    for case_name, file_name in (
        ("No time windows", "route_no_windows.png"),
        ("Wide time windows", "route_wide_windows.png"),
        ("Tight feasible", "route_tight_feasible.png"),
    ):
        scenario, variables, schedule_rows = solved_cases[case_name]
        service_times = {
            row["customer"]: row["service_start"] for row in schedule_rows
        }
        route_file = output_directory / file_name
        plot_solution(
            scenario,
            variables,
            vehicle_types,
            route_file,
            show=show,
            show_time_windows=scenario["use_time_constraints"],
            service_times=service_times,
        )
        route_files[case_name] = route_file

    return comparison_file, route_files


# -----------------------------------------------------------------------------
# 6. MAIN WORKFLOW
# -----------------------------------------------------------------------------

def main():
    """Run Scenario 3 and save its presentation-ready outputs."""
    show_plots = "--show" in sys.argv
    scenarios, vehicle_types = load_inputs()
    output_directory = get_output_directory("scenario3")

    print("Running time-window comparison...")
    results, solved_cases = run_time_window_experiment(
        scenarios, vehicle_types
    )
    report = build_report(results, solved_cases)
    report_file = save_report(report, output_directory)
    comparison_file, route_files = save_summary_plots(
        results,
        solved_cases,
        vehicle_types,
        output_directory,
        show_plots,
    )

    print("\n" + report)
    print(f"Results saved in: {report_file}")
    print(f"Comparison plot saved in: {comparison_file}")
    for case_name, route_file in route_files.items():
        print(f"{case_name} route saved in: {route_file}")


if __name__ == "__main__":
    main()
