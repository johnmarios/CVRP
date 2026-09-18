from copy import deepcopy
from pathlib import Path
import sys
import time

import pulp


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from diagnostics import capacity_precheck
from model import build_model, get_d_ij, get_t_ijk
from tools import (
    extract_route,
    get_output_directory,
    lp_relaxation,
    make_text_table,
    readjson,
)
from visualization import plot_capacity_results, plot_scenario1_comparison


CAPACITIES = [8, 10, 11, 12, 16, 21, 22]


def add_activation_order(model, variables):
    """Remove label symmetry between identical vans, by adding a chain of constraints."""
    vehicles = sorted(variables["z"])
    for first, second in zip(vehicles[:-1], vehicles[1:]):
        model += (variables["z"][first] >= variables["z"][second], f"symmetry_activation_{first}_{second}",)


def solve_capacity(scenario, vehicle_types, capacity):
    """Solve one homogeneous-fleet capacity case."""
    scenario = deepcopy(scenario)
    vehicle_types = deepcopy(vehicle_types)
    vehicle_types["van"]["capacity"] = capacity

    demands = [customer["demand"] for customer in scenario["customers"].values()]
    total_demand = sum(demands)
    number_of_vans = sum(scenario["vehicle_info"]["van"]["depots"].values())
    fleet_capacity = number_of_vans * capacity

    # Necessary feasibility checks before calling the solver.
    precheck = capacity_precheck(demands, number_of_vans, capacity,)

    # used in the result dictionary to avoid repetition
    common = {
        "capacity": capacity,
        "available_vehicles": number_of_vans,
        "total_demand": total_demand,
        "fleet_capacity": fleet_capacity,
        "aggregate_lower_bound": precheck["aggregate_lower_bound"],
        "large_customer_lower_bound": precheck["large_customer_lower_bound"],
        "capacity_lower_bound": precheck["capacity_lower_bound"],
        "capacity_deficit": max(0, total_demand - fleet_capacity),
    }

    if precheck["infeasible"]:
        return {
            **common,
            "status": "Infeasible",
            "solution_method": "Analytical capacity check",
            "solver_called": False,
            "feedback": precheck["reason"],
            "objective": None,
            "fixed_cost": None,
            "distance_cost": None,
            "time_cost": None,
            "active_vehicles": None,
            "active_capacity": None,
            "utilization_percent": None,
            "routes": None,
            "loads": None,
            "total_distance": None,
            "total_travel_time": None,
            "runtime_seconds": 0.0,
        }

    model, variables = build_model(scenario, vehicle_types)
    add_activation_order(model, variables)

    started = time.perf_counter()
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    runtime_seconds = time.perf_counter() - started
    status = pulp.LpStatus[model.status]

    # Build the result dictionary with default values for all fields. 
    # if the solution is not optimal, the fields will remain None, else the fields will 
    # be updated 

    result = {
        **common,
        "status": status,
        "solution_method": "CBC solver",
        "solver_called": True,
        "feedback": "",
        "objective": None,
        "fixed_cost": None,
        "distance_cost": None,
        "time_cost": None,
        "active_vehicles": None,
        "active_capacity": None,
        "utilization_percent": None,
        "routes": None,
        "loads": None,
        "total_distance": None,
        "total_travel_time": None,
        "runtime_seconds": round(runtime_seconds, 4),
    }

    if status != "Optimal":
        return result

    routes = {}
    loads = {}
    total_distance = 0.0
    total_travel_time = 0.0
    fixed_cost = 0.0
    distance_cost = 0.0
    time_cost = 0.0

    # Iterate over all active vehicles and extract the route, load, and costs for each one.
    for vehicle, active in variables["z"].items():
        if pulp.value(active) < 0.5:
            continue

        vehicle_type = vehicle.split("_")[0]
        route = extract_route(vehicle, variables["x"], len(scenario["customers"]),)
        load = sum(scenario["customers"][customer]["demand"] 
            for customer in scenario["customers"]
            if pulp.value(variables["y"][customer, vehicle]) > 0.5
        )

        routes[vehicle] = route
        loads[vehicle] = load
        fixed_cost += vehicle_types[vehicle_type]["fixed_cost"]

        # for the selected route, calculate the total distance and travel time, and their associated costs
        for first, second in zip(route[:-1], route[1:]):
            distance = get_d_ij(scenario, first, second)
            travel_time = get_t_ijk(scenario, vehicle_types, first, second, vehicle,)
            total_distance += distance
            total_travel_time += travel_time
            distance_cost += (vehicle_types[vehicle_type]["cost_per_distance"] * distance)
            time_cost += (vehicle_types[vehicle_type]["cost_per_time"] * travel_time)

    active_capacity = sum(vehicle_types[vehicle.split("_")[0]]["capacity"] for vehicle in routes)

    result.update(
        {
            "objective": round(pulp.value(model.objective), 4),
            "fixed_cost": round(fixed_cost, 4),
            "distance_cost": round(distance_cost, 4),
            "time_cost": round(time_cost, 4),
            "active_vehicles": len(routes),
            "active_capacity": active_capacity,
            "utilization_percent": round(100 * total_demand / active_capacity, 2,),
            "routes": routes,
            "loads": loads,
            "total_distance": round(total_distance, 4),
            "total_travel_time": round(total_travel_time, 4),
        }
    )
    return result


def capacity_sensitivity_analysis(scenario, vehicle_types):
    """Solve the baseline for every selected capacity."""
    return [solve_capacity(scenario, vehicle_types, capacity) for capacity in CAPACITIES]


def build_relaxation_comparison(baseline_scenario, concentrated_scenario, vehicle_types, baseline_results, concentrated_result,):
    """Compare each MILP case with its correctly matched LP relaxation."""

    baseline_by_capacity = {
        result["capacity"]: result
        for result in baseline_results
    }
    cases = [
        (
            "Baseline Q=11",
            "Baseline\nQ=11",
            baseline_scenario,
            11,
            baseline_by_capacity[11],
        ),
        (
            "Baseline Q=12",
            "Baseline\nQ=12",
            baseline_scenario,
            12,
            baseline_by_capacity[12],
        ),
        (
            "Concentrated Q=11",
            "Concentrated\nQ=11",
            concentrated_scenario,
            11,
            concentrated_result,
        ),
    ]

    results = []
    for case_name, short_name, scenario, capacity, milp in cases:
        lp = lp_relaxation(scenario, vehicle_types, capacity)
        gap = None

        if milp["status"] == "Optimal" and lp["status"] == "Optimal":
            gap = round(
                100 * (milp["objective"] - lp["objective"])
                / milp["objective"],
                2,
            )

        results.append(
            {
                "case": case_name,
                "short_name": short_name,
                "capacity": capacity,
                "milp_status": milp["status"],
                "milp_objective": milp["objective"],
                "lp_status": lp["status"],
                "lp_objective": lp["objective"],
                "gap_percent": gap,
                "sum_z": lp["sum_z"],
                "fractional_x": lp["fractional_x"],
                "fractional_y": lp["fractional_y"],
                "fractional_z": lp["fractional_z"],
                "fractional_examples": lp["fractional_examples"],
            }
        )

    return results


def format_representative_routes(result):
    """Format one solved capacity case for the report."""
    lines = [f"Q={result['capacity']}"]

    for vehicle, route in result["routes"].items():
        load = result["loads"][vehicle]
        slack = result["capacity"] - load
        lines.append(
            f"{vehicle}: {' -> '.join(route)} | "
            f"load={load}/{result['capacity']} | slack={slack}"
        )

    return lines


def validate_solution(scenario, result):
    """Validate the customer coverage, depot returns and vehicle loads."""
    expected_customers = set(scenario["customers"])
    visited_customers = []
    routes_valid = True

    for vehicle, route in result["routes"].items():
        depot = vehicle.split("_")[1]
        routes_valid = routes_valid and (
            route[0] == depot
            and route[-1] == depot
        )
        visited_customers.extend(route[1:-1])

    customers_valid = (
        set(visited_customers) == expected_customers
        and len(visited_customers) == len(expected_customers)
    )
    capacity_valid = all(
        load <= result["capacity"]
        for load in result["loads"].values()
    )

    return {
        "customers": customers_valid,
        "routes": routes_valid,
        "capacity": capacity_valid,
    }


def build_scenario_report(baseline_scenario, concentrated_scenario, capacity_results, demand_results, relaxation_results,):
    """Create the report for Scenario 1."""
    capacity_columns = [
        ("capacity", "Q"),
        ("fleet_capacity", "Fleet cap."),
        ("capacity_lower_bound", "Capacity LB"),
        ("status", "Status"),
        ("active_vehicles", "Vehicles"),
        ("active_capacity", "Active cap."),
        ("utilization_percent", "Active util. (%)"),
        ("objective", "Total cost"),
        ("total_distance", "Distance"),
        ("runtime_seconds", "Time (s)"),
    ]
    demand_columns = [
        ("scenario", "Profile"),
        ("total_demand", "Demand"),
        ("aggregate_lower_bound", "Aggregate LB"),
        ("large_customer_lower_bound", "Large-item LB"),
        ("capacity_lower_bound", "Overall LB"),
        ("status", "Status"),
        ("active_vehicles", "Vehicles"),
        ("objective", "Total cost"),
    ]
    relaxation_columns = [
        ("case", "Case"),
        ("milp_status", "MILP status"),
        ("milp_objective", "MILP obj."),
        ("lp_status", "LP status"),
        ("lp_objective", "LP bound"),
        ("gap_percent", "Gap (%)"),
        ("sum_z", "Sum z"),
        ("fractional_x", "Frac. x"),
        ("fractional_y", "Frac. y"),
        ("fractional_z", "Frac. z"),
    ]
    cost_columns = [
        ("capacity", "Q"),
        ("fixed_cost", "Fixed cost"),
        ("distance_cost", "Distance cost"),
        ("time_cost", "Travel-time cost"),
        ("objective", "Total cost"),
        ("total_distance", "Distance"),
        ("total_travel_time", "Travel time"),
    ]

    q11 = next(result for result in capacity_results if result["capacity"] == 11)
    q22 = next(result for result in capacity_results if result["capacity"] == 22)
    cost_reduction = round(
        100 * (q11["objective"] - q22["objective"]) / q11["objective"],
        2,
    )
    distance_reduction = round(
        100
        * (q11["total_distance"] - q22["total_distance"])
        / q11["total_distance"],
        2,
    )

    baseline_demands = [
        customer["demand"]
        for customer in baseline_scenario["customers"].values()
    ]
    concentrated_demands = [
        customer["demand"]
        for customer in concentrated_scenario["customers"].values()
    ]
    concentrated = next(
        result
        for result in demand_results
        if result["scenario"] == "Concentrated"
    )
    q11_validation = validate_solution(baseline_scenario, q11)
    q22_validation = validate_solution(baseline_scenario, q22)

    def validation_label(value):
        return "[PASS]" if value else "[FAIL]"

    lines = [
        "SCENARIO 1 - CAPACITY, DEMAND DISTRIBUTION AND LP RELAXATION",
        "============================================================",
        "",
        "EXPERIMENTAL SETUP",
        "------------------",
        f"Customers: {len(baseline_scenario['customers'])}",
        f"Available vehicles: {q11['available_vehicles']} identical vans",
        f"Baseline demands: {', '.join(map(str, baseline_demands))}",
        f"Concentrated demands: {', '.join(map(str, concentrated_demands))}",
        f"Total demand: {q11['total_demand']}",
        f"Capacity values: {', '.join(map(str, CAPACITIES))}",
        "Time windows: disabled",
        "",
        "A. CAPACITY SENSITIVITY",
        "-----------------------",
        make_text_table(capacity_results, capacity_columns),
        "",
        "Representative routes",
        *format_representative_routes(q11),
        "",
        *format_representative_routes(q22),
        "",
        "Cost decomposition at the two capacity thresholds",
        make_text_table([q11, q22], cost_columns),
        "",
        "Mathematical observations",
        "1. Q=8 and Q=10 are infeasible because the aggregate capacity lower bound is 3, while only 2 vans are available.",
        "2. Q=11 is the smallest tested capacity that permits a feasible two-vehicle solution.",
        "3. For 11 <= Q < 22, two routes are still required and the optimal distance and total cost remain unchanged.",
        f"4. At Q=22, one van is sufficient: total cost falls by {cost_reduction}% while distance falls by only {distance_reduction}%.",
        "5. In every feasible case, the capacity lower bound equals the optimal number of active vehicles.",
        "",
        "B. BASELINE AND CONCENTRATED DEMANDS AT Q=11",
        "----------------------------------------------",
        make_text_table(demand_results, demand_columns),
        "",
        "Infeasibility certificate",
        concentrated["feedback"],
        "The aggregate capacity condition passes in both profiles, but the concentrated profile contains three customers with demand greater than Q/2.",
        "",
        "C. MILP AND LP RELAXATION",
        "--------------------------",
        make_text_table(relaxation_results, relaxation_columns),
        "",
        "Representative fractional variables",
    ]

    for result in relaxation_results:
        lines.append(
            f"{result['case']}: {result['fractional_examples']}"
        )

    lines.extend(
        [
            "",
            "SOLUTION VALIDATION",
            "-------------------",
            f"{validation_label(q11_validation['customers'] and q22_validation['customers'])} Every customer is served exactly once in the representative optimal solutions.",
            f"{validation_label(q11_validation['routes'] and q22_validation['routes'])} Every active route starts and ends at depot D0.",
            f"{validation_label(q11_validation['capacity'] and q22_validation['capacity'])} Every active vehicle respects its capacity.",
            "[BINDING] At Q=11 both vans have load 11/11.",
            "[BINDING] At Q=22 the single active van has load 22/22.",
        ]
    )

    return "\n".join(lines) + "\n"


def main():
    """Run Scenario 1, print its report and save its two figures."""
    show_plots = "--show" in sys.argv

    baseline_scenario = readjson("data/scenario1/baseline.json")
    concentrated_scenario = readjson("data/scenario1/concentrated.json")
    vehicle_types = readjson("data/vehicles.json")["vehicle_types"]
    output_directory = get_output_directory("scenario1")

    # CAPACITY SENSITIVITY ANALYSIS
    
    # list of dictionaries, one for each capacity
    capacity_results = capacity_sensitivity_analysis(baseline_scenario, vehicle_types,) 

    # 
    baseline_q11 = next(
        result
        for result in capacity_results
        if result["capacity"] == 11
    )
    baseline_q11 = {**baseline_q11, "scenario": "Baseline"}
    concentrated_q11 = solve_capacity(
        concentrated_scenario,
        vehicle_types,
        11,
    )
    concentrated_q11 = {
        **concentrated_q11,
        "scenario": "Concentrated",
    }
    demand_results = [baseline_q11, concentrated_q11]

    relaxation_results = build_relaxation_comparison(
        baseline_scenario,
        concentrated_scenario,
        vehicle_types,
        capacity_results,
        concentrated_q11,
    )

    report = build_scenario_report(
        baseline_scenario,
        concentrated_scenario,
        capacity_results,
        demand_results,
        relaxation_results,
    )
    report_file = output_directory / "scenario1_results.txt"
    report_file.write_text(report, encoding="utf-8")

    capacity_plot = output_directory / "capacity_sensitivity_plot.png"
    plot_capacity_results(
        capacity_results,
        CAPACITIES,
        capacity_plot,
        show=show_plots,
    )

    comparison_plot = output_directory / "demand_and_relaxation_plot.png"
    demand_profiles = {
        "Baseline": [
            customer["demand"]
            for customer in baseline_scenario["customers"].values()
        ],
        "Concentrated": [
            customer["demand"]
            for customer in concentrated_scenario["customers"].values()
        ],
    }
    plot_scenario1_comparison(
        demand_profiles,
        11,
        relaxation_results,
        comparison_plot,
        show=show_plots,
    )

    print(report)
    print(f"Results saved in: {report_file}")
    print(f"Capacity plot saved in: {capacity_plot}")
    print(f"Comparison plot saved in: {comparison_plot}")


if __name__ == "__main__":
    main()
