from copy import deepcopy
from pathlib import Path
import sys

import pulp


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from diagnostics import capacity_precheck
from model import build_model
from tools import (
    add_symmetry_breaking,
    calculate_route_costs,
    extract_routes_and_loads,
    format_loads,
    format_routes,
    get_output_directory,
    get_vehicle_count,
    lp_relaxation,
    make_text_table,
    readjson,
    save_report,
    solve_model,
)
from visualization import (
    plot_capacity_results,
    plot_demand_distribution,
    plot_relaxation_results,
    plot_solution,
    plot_scenario_demo,
)


CAPACITIES = [8, 10, 11, 12, 16, 21, 22]
TEST_CAPACITY = 11


# INPUT DATA

def load_inputs():
    """Load the two demand scenarios and the common vehicle data."""
    baseline = readjson("data/scenario1/baseline.json")
    concentrated = readjson("data/scenario1/concentrated.json")
    vehicle_types = readjson("data/vehicles.json")["vehicle_types"]
    return baseline, concentrated, vehicle_types


def get_demands(scenario):
    """Return the customer demands of a scenario."""
    return [customer["demand"] for customer in scenario["customers"].values()]


def prepare_case(scenario, vehicle_types, capacity):
    """Copy the input data and change only the van capacity."""
    scenario = deepcopy(scenario)
    vehicle_types = deepcopy(vehicle_types)
    vehicle_types["van"]["capacity"] = capacity
    return scenario, vehicle_types



# SOLVE ONE CAPACITY CASE

def create_result(capacity, demands, number_of_vans, check):
    """Create an empty result dictionary with all common information."""
    return {
        "capacity": capacity,
        "total_demand": sum(demands),
        "available_vehicles": number_of_vans,
        "fleet_capacity": number_of_vans * capacity,
        "aggregate_lower_bound": check["aggregate_lower_bound"],
        "large_customer_lower_bound": check["large_customer_lower_bound"],
        "capacity_lower_bound": check["capacity_lower_bound"],
        "status": "Infeasible" if check["infeasible"] else "Not solved",
        "feedback": check["reason"],
        "objective": None,
        "fixed_cost": None,
        "distance_cost": None,
        "time_cost": None,
        "active_vehicles": None,
        "active_capacity": None,
        "utilization_percent": None,
        "total_distance": None,
        "total_travel_time": None,
        "runtime_seconds": 0.0,
        "routes": {},
        "loads": {},
    }


def complete_result(result, model, scenario, vehicle_types, variables):
    """Add routes, loads and cost information to an optimal result."""
    routes, loads = extract_routes_and_loads(scenario, variables)
    costs = calculate_route_costs(scenario, vehicle_types, routes)
    active_capacity = sum(vehicle_types[vehicle.split("_")[0]]["capacity"] for vehicle in routes)

    objective = pulp.value(model.objective)
    util_percent = int(100 * result["total_demand"] / active_capacity)

    result.update(costs)
    result.update(
        {
            "objective": round(objective, 4),
            "active_vehicles": len(routes),
            "active_capacity": active_capacity,
            "utilization_percent": round(util_percent, 2),
            "routes": routes,
            "loads": loads,
        }
    )
    return result


def solve_capacity(base_scenario, base_vehicle_types, capacity):
    """Run the complete workflow for one capacity value."""

    # prepare scenario for the requested Q
    scenario, vehicle_types = prepare_case(base_scenario, base_vehicle_types, capacity)
    demands = get_demands(scenario)
    number_of_vans = get_vehicle_count(scenario, "van")

    # run some capacity checks for infeasibility before calling the solver
    check = capacity_precheck(demands, number_of_vans, capacity)

    # create a result dictionary with all common information
    result = create_result(capacity, demands, number_of_vans, check)

    # A failed necessary condition proves infeasibility without a solver call.
    if check["infeasible"]:
        return result

    model, variables = build_model(scenario, vehicle_types)
    add_symmetry_breaking(model, variables, scenario)
    result["status"], result["runtime_seconds"] = solve_model(model)

    if result["status"] == "Optimal":
        complete_result(result, model, scenario, vehicle_types, variables)

    # in case of infeasibility 
    return result


#EXPERIMENTS

def run_capacity_experiment(scenario, vehicle_types):
    """Solve the baseline scenario for every selected capacity."""
    results = []

    for capacity in CAPACITIES:
        result = solve_capacity(scenario, vehicle_types, capacity)
        results.append(result)

    return results


def find_capacity_result(results, capacity):
    """Return the capacity result that corresponds to Q, by extracting it from the results list."""
    return next(result for result in results if result["capacity"] == capacity)


def run_demand_experiment(baseline_result, concentrated_scenario, vehicle_types,):
    """Compare baseline and concentrated demands at Q=11."""
    concentrated_result = solve_capacity(concentrated_scenario, vehicle_types, TEST_CAPACITY,)

    # combine the two results in one list for the report and plots
    demand_results = [
        {**baseline_result, "scenario": "Baseline"},
        {**concentrated_result, "scenario": "Concentrated"},
    ]
    return demand_results, concentrated_result


def calculate_lp_gap(milp_result, lp_result):
    """Return the percentage gap when both problems are optimal."""

    if milp_result["status"] != "Optimal" or lp_result["status"] != "Optimal":
        return None

    return round(100 * (milp_result["objective"] - lp_result["objective"]) / milp_result["objective"], 2,)


def create_relaxation_row(name, short_name, milp_result, lp_result):
    """Combine the MILP and LP results of one case in one table row."""
    return {
        "case": name,
        "short_name": short_name,
        "milp_status": milp_result["status"],
        "milp_objective": milp_result["objective"],
        "lp_status": lp_result["status"],
        "lp_objective": lp_result["objective"],
        "gap_percent": calculate_lp_gap(milp_result, lp_result),
        "sum_z": lp_result["sum_z"],
        "fractional_x": lp_result["fractional_x"],
        "fractional_y": lp_result["fractional_y"],
        "fractional_z": lp_result["fractional_z"],
        "fractional_examples": lp_result["fractional_examples"],
    }


def run_relaxation_experiment(baseline_scenario, concentrated_scenario, vehicle_types, capacity_results, concentrated_result,):
    """Compare three MILP cases with their matching LP relaxations."""
    cases = [
        (
            "Baseline Q=11", "Baseline\nQ=11", baseline_scenario, 11,
            find_capacity_result(capacity_results, 11),
        ),
        (
            "Baseline Q=12", "Baseline\nQ=12", baseline_scenario, 12,
            find_capacity_result(capacity_results, 12),
        ),
        (
            "Concentrated Q=11", "Concentrated\nQ=11",
            concentrated_scenario, 11, concentrated_result,
        ),
    ]

    results = []
    for name, short_name, scenario, capacity, milp_result in cases:
        lp_result = lp_relaxation(scenario, vehicle_types, capacity)
        results.append(
            create_relaxation_row(name, short_name, milp_result, lp_result)
        )

    return results


# TEXT REPORT

# build the report sections and join them in one string

def build_setup_section(baseline, concentrated, number_of_vans):
    """Describe the common experimental setup."""
    report_str = (
        "EXPERIMENTAL SETUP\n"
        "------------------\n"
        f"Customers: {len(baseline['customers'])}\n"
        f"Available vehicles: {number_of_vans} identical vans\n"
        f"Baseline demands: {', '.join(map(str, get_demands(baseline)))}\n"
        f"Concentrated demands: {', '.join(map(str, get_demands(concentrated)))}\n"
        f"Total demand: {sum(get_demands(baseline))}\n"
        f"Capacity values: {', '.join(map(str, CAPACITIES))}"
    )
    return report_str


def build_capacity_section(capacity_results):
    """Create the capacity table, representative routes and cost table."""

    # for capacity table
    columns = [
        ("capacity", "Q"),
        ("fleet_capacity", "Fleet cap."),
        ("capacity_lower_bound", "Capacity LB"),
        ("status", "Status"),
        ("active_vehicles", "Vehicles"),
        ("utilization_percent", "Util. (%)"),
        ("objective", "Total cost"),
        ("total_distance", "Distance"),
    ]

    # for cost decomposition table (q11 and q22 only)
    cost_columns = [
        ("capacity", "Q"),
        ("fixed_cost", "Fixed cost"),
        ("distance_cost", "Distance cost"),
        ("time_cost", "Time cost"),
        ("objective", "Total cost"),
    ]

    q11 = find_capacity_result(capacity_results, 11)
    q22 = find_capacity_result(capacity_results, 22)

    report_str = (
        "A. CAPACITY SENSITIVITY\n"
        "-----------------------\n"
        + make_text_table(capacity_results, columns) + "\n"
        "\n"
        "Representative routes\n"
        f"Q=11 routes: {format_routes(q11['routes'])}\n"
        f"Q=11 loads:  {format_loads(q11['loads'])}\n"
        "\n"
        f"Q=22 routes: {format_routes(q22['routes'])}\n"
        f"Q=22 loads:  {format_loads(q22['loads'])}\n"
        "\n"
        "Cost decomposition\n"
        + make_text_table([q11, q22], cost_columns)
    )

    return report_str


def build_demand_section(demand_results):
    """Create the demand-distribution table and infeasibility feedback."""
    columns = [
        ("scenario", "Profile"),
        ("aggregate_lower_bound", "Aggregate LB"),
        ("large_customer_lower_bound", "Large-item LB"),
        ("capacity_lower_bound", "Overall LB"),
        ("status", "Status"),
        ("objective", "Total cost"),
    ]
    concentrated = demand_results[1]

    report_str = (
        "B. DEMAND DISTRIBUTION AT Q=11\n"
        "------------------------------\n"
        + make_text_table(demand_results, columns) + "\n"
        "\n"
        "Infeasibility feedback\n"
        f"{concentrated['feedback']}"
    )
    return report_str



def build_relaxation_section(relaxation_results):
    """Create the MILP-LP table and show representative fractional values."""
    columns = [
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
    examples = "\n".join(
        f"{result['case']}: {result['fractional_examples']}"
        for result in relaxation_results
    )

    report_str = (
        "C. MILP AND LP RELAXATION\n"
        "--------------------------\n"
        + make_text_table(relaxation_results, columns) + "\n"
        "\n"
        "Representative fractional variables\n"
        f"{examples}"
    )
    return report_str

# build report from sections and return it as a single string
def build_report(
    baseline,
    concentrated,
    capacity_results,
    demand_results,
    relaxation_results,
):
    """Join the setup and the three experiment sections in one report."""
    title = ("SCENARIO 1 - CAPACITY, DEMAND DISTRIBUTION AND LP RELAXATION\n"
        "============================================================")
    sections = [
        title,
        build_setup_section(
            baseline,
            concentrated,
            get_vehicle_count(baseline, "van"),
        ),
        build_capacity_section(capacity_results),
        build_demand_section(demand_results),
        build_relaxation_section(relaxation_results),
    ]
    report_str = "\n\n".join(sections) + "\n"
    return report_str


# outputs

def save_plots(
    baseline,
    concentrated,
    capacity_results,
    relaxation_results,
    output_directory,
    show_plots,
):
    """Save the three Scenario 1 plots and return their paths."""
    capacity_plot = output_directory / "capacity_sensitivity_plot.png"

    plot_capacity_results(
        capacity_results,
        CAPACITIES,
        capacity_plot,
        show=show_plots,
    )

    demand_profiles = {
        "Baseline": get_demands(baseline),
        "Concentrated": get_demands(concentrated),
    }
    demand_plot = output_directory / "demand_distribution_plot.png"
    plot_demand_distribution(
        demand_profiles,
        TEST_CAPACITY,
        demand_plot,
        show=show_plots,
    )

    relaxation_plot = output_directory / "lp_relaxation_plot.png"
    plot_relaxation_results(
        relaxation_results,
        relaxation_plot,
        show=show_plots,
    )
    return capacity_plot, demand_plot, relaxation_plot



# MAIN WORKFLOW

def main():

    """Run the experiments and save the presentation-ready outputs."""
    show_plots = "--show" in sys.argv

    # set up the input data and output directory
    baseline, concentrated, vehicle_types = load_inputs()
    output_directory = get_output_directory("scenario1")

    # plot_scenario_demo(baseline)
    # pass

    print("Running capacity sensitivity...")
    capacity_results = run_capacity_experiment(baseline, vehicle_types)

    # get the result from the list 
    baseline_q11 = find_capacity_result(capacity_results, 11)

    # demand results for Q=11 (baseline and concentrated)
    demand_results, concentrated_q11 = run_demand_experiment(baseline_q11, concentrated, vehicle_types,)

    # relaxation results for three cases
    print("Running LP relaxations...")
    relaxation_results = run_relaxation_experiment(baseline, concentrated, vehicle_types, capacity_results, concentrated_q11,)

    # build and save report 
    report = build_report(
        baseline,
        concentrated,
        capacity_results,
        demand_results,
        relaxation_results,
    )
    report_file = save_report(
        report, output_directory, "scenario1_results.txt"
    )

    # save the capacity, demand-distribution and LP-relaxation plots
    capacity_plot, demand_plot, relaxation_plot = save_plots(
        baseline,
        concentrated,
        capacity_results,
        relaxation_results,
        output_directory,
        show_plots,
    )

    print("\n" + report)
    print(f"Results saved in: {report_file}")
    print(f"Capacity plot saved in: {capacity_plot}")
    print(f"Demand plot saved in: {demand_plot}")
    print(f"LP relaxation plot saved in: {relaxation_plot}")

    # plot milp baseline solution for Q=11 and Q=22 and concentrated solution for Q=11
    plot_solution(
        baseline,
        baseline_q11["routes"],
        vehicle_types,
        output_directory / "baseline_q11_solution.png",
        show=show_plots,
    )
    plot_solution(
        concentrated,
        concentrated_q11["routes"],
        vehicle_types,
        output_directory / "concentrated_q11_solution.png",
        show=show_plots,
    )

    plot_solution(
        baseline,
        find_capacity_result(capacity_results, 22)["routes"],
        vehicle_types,
        output_directory / "baseline_q22_solution.png",
        show=show_plots,
    )

    plot_solution(
        concentrated,
        find_capacity_result(capacity_results, 22)["routes"],
        vehicle_types,
        output_directory / "concentrated_q22_solution.png",
        show=show_plots,
    )


if __name__ == "__main__":
    main()
