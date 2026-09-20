from copy import deepcopy
from pathlib import Path
import sys

import pulp


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from model import build_model
from tools import (
    add_symmetry_breaking,
    calculate_route_costs,
    extract_routes_and_loads,
    find_case,
    format_loads,
    format_routes,
    get_output_directory,
    make_text_table,
    readjson,
    save_report,
    solve_model,
)
from visualization import (
    plot_solution,
    plot_truck_cost_sensitivity,
)


FLEET_CASES = {
    "Two vans": "data/scenario2/two_vans.json",
    "One truck": "data/scenario2/one_truck.json",
    "Mixed fleet": "data/scenario2/mixed_fleet.json",
    "Free choice": "data/scenario2/free_choice.json",
}

TRUCK_FIXED_COSTS = [32, 28, 24, 22, 21, 20, 18]
TRUCK_DISTANCE_COSTS = [0.95, 0.80, 0.70, 0.60, 0.55, 0.53, 0.52, 0.50]
SWITCH_TEST_FIXED_COST = 21.0



# input data and fleet availability

def load_inputs():
    """Load the four fleet scenarios and the common vehicle parameters."""
    scenarios = {
        case_name: readjson(file_name)
        for case_name, file_name in FLEET_CASES.items()
    }
    vehicle_types = readjson("data/vehicles.json")["vehicle_types"]
    return scenarios, vehicle_types


def get_fleet_counts(scenario):
    """Return the number of available vehicles of each type."""
    return {
        vehicle_type: sum(information["depots"].values())
        for vehicle_type, information in scenario["vehicle_info"].items()
    }


def fleet_text(counts):
    """Convert vehicle counts to a short readable description, vehicle_type=count."""
    selected = [
        f"{vehicle_type}={count}"
        for vehicle_type, count in counts.items()
        if count > 0
    ]
    return ", ".join(selected) or "none"


# solve one fleet case and extract the results
def create_result(case_name, scenario, vehicle_types):
    """Create an empty result containing the common case information."""
    # available vehicles, capacity and total demand
    available_counts = get_fleet_counts(scenario)
    available_capacity = sum(
        count * vehicle_types[vehicle_type]["capacity"]
        for vehicle_type, count in available_counts.items()
    )
    total_demand = sum(customer["demand"] for customer in scenario["customers"].values())

    return {
        "case": case_name,
        "available_fleet": fleet_text(available_counts),
        "selected_fleet": None,
        "status": "Not solved",
        "active_vehicles": None,
        "total_demand": total_demand,
        "available_capacity": available_capacity,
        "active_capacity": None,
        "utilization_percent": None,
        "total_distance": None,
        "total_travel_time": None,
        "fixed_cost": None,
        "distance_cost": None,
        "time_cost": None,
        "variable_cost": None,
        "objective": None,
        "runtime_seconds": 0.0,
        "routes": {},
        "loads": {},
    }


def count_active_vehicle_types(routes, vehicle_types):
    """Count how many active vehicles belong to each type."""
    counts = {vehicle_type: 0 for vehicle_type in vehicle_types}
    for vehicle in routes:
        counts[vehicle.split("_")[0]] += 1
    return counts


def complete_result(result, model, scenario, vehicle_types, variables):
    """Add the selected fleet, routes and costs to an optimal result."""
    routes, loads = extract_routes_and_loads(scenario, variables)
    active_counts = count_active_vehicle_types(routes, vehicle_types)
    costs = calculate_route_costs(
        scenario,
        vehicle_types,
        routes,
        include_variable_cost=True,
    )
    active_capacity = sum(
        vehicle_types[vehicle.split("_")[0]]["capacity"]
        for vehicle in routes
    )

    result.update(costs)
    result.update(
        {
            "selected_fleet": fleet_text(active_counts),
            "active_vehicles": len(routes),
            "active_capacity": active_capacity,
            "utilization_percent": round(100 * result["total_demand"] / active_capacity, 2),
            "objective": round(pulp.value(model.objective), 4),
            "routes": routes,
            "loads": loads,
        }
    )
    return result


def solve_fleet_case(case_name, scenario, vehicle_types):
    """Build, solve and summarize one fleet-composition case."""
    model, variables = build_model(scenario, vehicle_types)
    add_symmetry_breaking(model, variables, scenario)
    result = create_result(case_name, scenario, vehicle_types)
    result["status"], result["runtime_seconds"] = solve_model(model)

    if result["status"] == "Optimal":
        complete_result(result, model, scenario, vehicle_types, variables)

    return result, variables


# fleet comparison 
def run_fleet_comparison(scenarios, vehicle_types):
    """Solve all four controlled fleet cases."""
    results = []
    solved_cases = {}

    for case_name, scenario in scenarios.items():
        result, variables = solve_fleet_case(
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


# -----------------------------------------------------------------------------
# 4. TRUCK COST SENSITIVITY
# -----------------------------------------------------------------------------

def cheaper_option(truck_cost, two_vans_cost):
    """Return the cheaper of the two controlled alternatives."""
    if truck_cost < two_vans_cost:
        return "One truck"
    if truck_cost > two_vans_cost:
        return "Two vans"
    return "Equal cost"


def calculate_critical_costs(two_vans, one_truck, vehicle_types):
    """Calculate the exact truck fixed- and distance-cost thresholds."""
    van_cost = two_vans["objective"]
    truck_distance = one_truck["total_distance"]
    truck_time = one_truck["total_travel_time"]
    truck = vehicle_types["truck"]
    time_cost = truck["cost_per_time"] * truck_time

    critical_fixed_cost = (
        van_cost - truck["cost_per_distance"] * truck_distance - time_cost
    )
    critical_distance_cost = (
        van_cost - truck["fixed_cost"] - time_cost
    ) / truck_distance

    return critical_fixed_cost, critical_distance_cost


def build_fixed_cost_results(two_vans, one_truck, vehicle_types):
    """Evaluate one-truck cost for selected fixed-cost values."""
    truck = vehicle_types["truck"]
    variable_cost = (
        truck["cost_per_distance"] * one_truck["total_distance"]
        + truck["cost_per_time"] * one_truck["total_travel_time"]
    )

    results = []
    for fixed_cost in TRUCK_FIXED_COSTS:
        truck_total_cost = fixed_cost + variable_cost
        results.append(
            {
                "truck_fixed_cost": fixed_cost,
                "truck_total_cost": round(truck_total_cost, 4),
                "two_vans_cost": two_vans["objective"],
                "difference": round(truck_total_cost - two_vans["objective"], 4),
                "cheaper_option": cheaper_option(
                    truck_total_cost, two_vans["objective"]
                ),
            }
        )
    return results


def build_distance_cost_results(two_vans, one_truck, vehicle_types):
    """Evaluate one-truck cost for selected distance-cost values."""
    truck = vehicle_types["truck"]
    fixed_and_time_cost = (
        truck["fixed_cost"]
        + truck["cost_per_time"] * one_truck["total_travel_time"]
    )

    results = []
    for distance_cost in TRUCK_DISTANCE_COSTS:
        truck_total_cost = (
            fixed_and_time_cost
            + distance_cost * one_truck["total_distance"]
        )
        results.append(
            {
                "truck_cost_per_distance": distance_cost,
                "truck_total_cost": round(truck_total_cost, 4),
                "two_vans_cost": two_vans["objective"],
                "difference": round(truck_total_cost - two_vans["objective"], 4),
                "cheaper_option": cheaper_option(
                    truck_total_cost, two_vans["objective"]
                ),
            }
        )
    return results


def validate_fleet_switch(free_choice_scenario, vehicle_types):
    """Resolve the free-choice MILP below the critical fixed cost."""
    changed_vehicle_types = deepcopy(vehicle_types)
    changed_vehicle_types["truck"]["fixed_cost"] = SWITCH_TEST_FIXED_COST
    scenario = deepcopy(free_choice_scenario)
    scenario["name"] = (
        f"Free choice with truck fixed cost {SWITCH_TEST_FIXED_COST:g}"
    )
    result, _ = solve_fleet_case(
        "Free choice after cost change",
        scenario,
        changed_vehicle_types,
    )
    return result


def run_cost_sensitivity(
    free_choice_scenario,
    vehicle_types,
    fleet_results,
):
    """Build the sensitivity tables and validate the predicted fleet switch."""
    two_vans = find_case(fleet_results, "Two vans")
    one_truck = find_case(fleet_results, "One truck")
    critical_fixed_cost, critical_distance_cost = calculate_critical_costs(
        two_vans, one_truck, vehicle_types
    )
    fixed_results = build_fixed_cost_results(
        two_vans, one_truck, vehicle_types
    )
    distance_results = build_distance_cost_results(
        two_vans, one_truck, vehicle_types
    )
    switch_result = validate_fleet_switch(
        free_choice_scenario, vehicle_types
    )

    return {
        "fixed_results": fixed_results,
        "distance_results": distance_results,
        "critical_fixed_cost": critical_fixed_cost,
        "critical_distance_cost": critical_distance_cost,
        "switch_result": switch_result,
    }


# -----------------------------------------------------------------------------
# 5. TEXT REPORT
# -----------------------------------------------------------------------------

def build_vehicle_table(vehicle_types):
    """Create rows containing the parameters of every vehicle type."""
    return [
        {
            "type": vehicle_type,
            "capacity": data["capacity"],
            "fixed_cost": data["fixed_cost"],
            "distance_cost": data["cost_per_distance"],
            "time_cost": data["cost_per_time"],
        }
        for vehicle_type, data in vehicle_types.items()
    ]


def build_setup_section(base_scenario, vehicle_types):
    """Describe the common demand and vehicle parameters."""
    columns = [
        ("type", "Type"),
        ("capacity", "Capacity"),
        ("fixed_cost", "Fixed cost"),
        ("distance_cost", "Cost/distance"),
        ("time_cost", "Cost/time"),
    ]
    total_demand = sum(
        customer["demand"]
        for customer in base_scenario["customers"].values()
    )

    report_str = (
        "EXPERIMENTAL SETUP\n"
        "------------------\n"
        f"Customers: {len(base_scenario['customers'])}\n"
        f"Total demand: {total_demand}\n"
        "Vehicle travel-time factors: disabled to isolate fleet capacity and cost\n"
        "\n"
        + make_text_table(build_vehicle_table(vehicle_types), columns)
    )
    return report_str


def build_fleet_section(fleet_results):
    """Create the fleet comparison table and route details."""
    columns = [
        ("case", "Case"),
        ("available_fleet", "Available fleet"),
        ("selected_fleet", "Selected fleet"),
        ("active_vehicles", "Vehicles"),
        ("active_capacity", "Active cap."),
        ("total_distance", "Distance"),
        ("fixed_cost", "Fixed cost"),
        ("distance_cost", "Distance cost"),
        ("time_cost", "Time cost"),
        ("objective", "Total cost"),
    ]
    route_lines = []
    for result in fleet_results:
        route_lines.extend(
            [
                f"{result['case']}:",
                f"Routes: {format_routes(result['routes'])}",
                f"Loads:  {format_loads(result['loads'])}",
                "",
            ]
        )

    route_text = "\n".join(route_lines).rstrip()
    report_str = (
        "A. FLEET COMPOSITION COMPARISON\n"
        "-------------------------------\n"
        + make_text_table(fleet_results, columns) + "\n"
        "\n"
        "ROUTE DETAILS\n"
        "-------------\n"
        f"{route_text}"
    )
    return report_str


def build_sensitivity_section(sensitivity):
    """Create the truck cost-sensitivity tables and critical-cost explanation."""
    fixed_columns = [
        ("truck_fixed_cost", "Truck fixed cost"),
        ("truck_total_cost", "Truck total cost"),
        ("two_vans_cost", "Two vans cost"),
        ("difference", "Truck - vans"),
        ("cheaper_option", "Cheaper option"),
    ]
    distance_columns = [
        ("truck_cost_per_distance", "Truck cost/distance"),
        ("truck_total_cost", "Truck total cost"),
        ("two_vans_cost", "Two vans cost"),
        ("difference", "Truck - vans"),
        ("cheaper_option", "Cheaper option"),
    ]
    switch = sensitivity["switch_result"]

    report_str = (
        "B. TRUCK COST SENSITIVITY\n"
        "-------------------------\n"
        f"Exact critical fixed cost: {sensitivity['critical_fixed_cost']:.4f}\n"
        f"Exact critical distance cost: {sensitivity['critical_distance_cost']:.4f}\n"
        "\n"
        "The fixed-cost threshold is obtained from:\n"
        "F_truck* = C_two_vans - c_distance_truck*D_truck - c_time_truck*T_truck.\n"
        "\n"
        "The distance-cost threshold is obtained from:\n"
        "c_distance_truck* = (C_two_vans - F_truck - c_time_truck*T_truck) / D_truck.\n"
        "\n"
        "FIXED-COST SENSITIVITY\n"
        + make_text_table(sensitivity['fixed_results'], fixed_columns) + "\n"
        "\n"
        "DISTANCE-COST SENSITIVITY\n"
        + make_text_table(sensitivity['distance_results'], distance_columns) + "\n"
        "\n"
        "Solver validation\n"
        f"After setting the truck fixed cost to {SWITCH_TEST_FIXED_COST:g}, "
        f"the free-choice MILP selects {switch['selected_fleet']} "
        f"with total cost {switch['objective']:.4f}."
    )
    return report_str


def build_report(base_scenario, vehicle_types, fleet_results, sensitivity):
    """Join the setup, fleet comparison and sensitivity analysis."""
    title = (
        "SCENARIO 2 - HETEROGENEOUS FLEET AND COST SENSITIVITY\n"
        "====================================================="
    )
    sections = [
        title,
        build_setup_section(base_scenario, vehicle_types),
        build_fleet_section(fleet_results),
        build_sensitivity_section(sensitivity),
    ]
    report_str = "\n\n".join(sections) + "\n"
    return report_str


# -----------------------------------------------------------------------------
# 6. OUTPUTS
# -----------------------------------------------------------------------------

def save_sensitivity_plot(sensitivity, output_directory, show_plots):
    """Save the truck-cost sensitivity plot."""
    sensitivity_plot = output_directory / "truck_cost_sensitivity.png"
    plot_truck_cost_sensitivity(
        sensitivity["fixed_results"],
        sensitivity["distance_results"],
        sensitivity["critical_fixed_cost"],
        sensitivity["critical_distance_cost"],
        sensitivity_plot,
        show=show_plots,
    )
    return sensitivity_plot


def save_route_plots(solved_cases, vehicle_types, output_directory, show_plots):
    """Save only the two route plots needed in the report."""
    route_files = {}
    for case_name, file_name in (
        ("Two vans", "route_two_vans.png"),
        ("One truck", "route_one_truck.png"),
        ("Free choice", "route_free_choice.png"),
        ("Mixed fleet", "route_mixed_fleet.png")
    ):
        scenario, variables = solved_cases[case_name]
        route_file = output_directory / file_name
        plot_solution(
            scenario,
            variables,
            vehicle_types,
            route_file,
            show=show_plots,
        )
        route_files[case_name] = route_file
    return route_files


# -----------------------------------------------------------------------------
# 7. MAIN WORKFLOW
# -----------------------------------------------------------------------------

def main():
    """Run Scenario 2 and save its presentation-ready outputs."""
    show_plots = "--show" in sys.argv
    scenarios, vehicle_types = load_inputs()
    output_directory = get_output_directory("scenario2")

    print("Running fleet comparison...")
    fleet_results, solved_cases = run_fleet_comparison(
        scenarios, vehicle_types
    )

    print("Running truck cost sensitivity...")
    sensitivity = run_cost_sensitivity(
        scenarios["Free choice"],
        vehicle_types,
        fleet_results,
    )

    report = build_report(
        scenarios["Free choice"],
        vehicle_types,
        fleet_results,
        sensitivity,
    )
    report_file = save_report(
        report, output_directory, "scenario2_results.txt"
    )
    sensitivity_plot = save_sensitivity_plot(
        sensitivity,
        output_directory,
        show_plots,
    )
    route_files = save_route_plots(
        solved_cases,
        vehicle_types,
        output_directory,
        show_plots,
    )

    print("\n" + report)
    print(f"Results saved in: {report_file}")
    print(f"Sensitivity plot saved in: {sensitivity_plot}")
    for case_name, route_file in route_files.items():
        print(f"{case_name} route saved in: {route_file}")

    


if __name__ == "__main__":
    main()
