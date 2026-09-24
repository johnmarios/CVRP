from copy import deepcopy
import json
from pathlib import Path
import time
import pulp


PROJECT_DIRECTORY = Path(__file__).resolve().parent


def readjson(filename: str) -> dict:
    """Read a JSON file and return its data."""
    filepath = PROJECT_DIRECTORY / filename

    with filepath.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_output_directory(scenario_name, experiment_name=None):
    """Return the output folder of a scenario or experiment."""
    output_directory = PROJECT_DIRECTORY / "outputs" / scenario_name

    if experiment_name:
        output_directory = output_directory / experiment_name

    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def extract_route(vehicle, x, number_of_customers):
    """Convert the selected arcs of a vehicle into an ordered route."""

    depot = vehicle.split("_")[1]
    # extract route from selected arcs
    successor = {i: j for (i, j, k), variable in x.items() if k == vehicle and pulp.value(variable) > 0.5}

    if depot not in successor:
        return []

    route = [depot]
    current = depot

    for _ in range(number_of_customers + 1):
        current = successor[current]
        route.append(current)

        if current == depot:
            return route

    raise RuntimeError(f"Invalid route for {vehicle}.")


def extract_routes_and_loads(scenario, variables):
    """Extract the route and carried load of every active vehicle."""
    routes = {}
    loads = {}

    for vehicle in sorted(variables["z"]):
        if pulp.value(variables["z"][vehicle]) < 0.5:
            continue

        routes[vehicle] = extract_route(
            vehicle,
            variables["x"],
            len(scenario["customers"]),
        )
        loads[vehicle] = sum(
            scenario["customers"][customer]["demand"]
            for customer in scenario["customers"]
            if pulp.value(variables["y"][customer, vehicle]) > 0.5
        )

    return routes, loads


def calculate_route_costs(
    scenario,
    vehicle_types,
    routes,
    include_variable_cost=False,
):
    """Calculate distance, travel time and the route-cost components."""
    from model import get_d_ij, get_t_ijk

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

    costs = {
        "fixed_cost": round(fixed_cost, 4),
        "distance_cost": round(distance_cost, 4),
        "time_cost": round(time_cost, 4),
        "total_distance": round(total_distance, 4),
        "total_travel_time": round(total_travel_time, 4),
    }
    if include_variable_cost:
        costs["variable_cost"] = round(distance_cost + time_cost, 4)
    return costs


def solve_model(model):
    """Solve a model with CBC and return its status and runtime."""
    start = time.perf_counter()
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    runtime = round(time.perf_counter() - start, 4)
    return pulp.LpStatus[model.status], runtime


def find_case(results, case_name):
    """Return one result using its case name."""
    return next(result for result in results if result["case"] == case_name)


def get_total_demand(scenario):
    """Return the sum of all customer demands."""
    return sum(
        customer["demand"] for customer in scenario["customers"].values()
    )


def get_fleet_counts(scenario):
    """Return the available number of vehicles of every type."""
    return {
        vehicle_type: sum(information["depots"].values())
        for vehicle_type, information in scenario["vehicle_info"].items()
    }


def get_vehicle_count(scenario, vehicle_type):
    """Return the available number of vehicles of one type."""
    return get_fleet_counts(scenario).get(vehicle_type, 0)


def count_active_vehicle_types(routes, vehicle_types):
    """Count the active vehicles of every type."""
    counts = {vehicle_type: 0 for vehicle_type in vehicle_types}
    for vehicle in routes:
        counts[vehicle.split("_")[0]] += 1
    return counts


def get_active_capacity(routes, vehicle_types):
    """Return the total capacity of the active vehicles."""
    return sum(
        vehicle_types[vehicle.split("_")[0]]["capacity"]
        for vehicle in routes
    )


def add_symmetry_breaking(model, variables, scenario):
    """Activate identical vehicles at each depot in a fixed order."""
    for vehicle_type, information in scenario["vehicle_info"].items():
        for depot in information["depots"]:
            prefix = f"{vehicle_type}_{depot}_"
            vehicles = sorted(
                vehicle
                for vehicle in variables["z"]
                if vehicle.startswith(prefix)
            )

            for first, second in zip(vehicles[:-1], vehicles[1:]):
                model += variables["z"][first] >= variables["z"][second]


def format_routes(routes):
    """Format the routes dictionary into a string representation."""
    if not routes:
        return "-"

    route_strings = []
    for vehicle, route in routes.items():
        route_str = " -> ".join(route)
        route_strings.append(f"{vehicle}: {route_str}")

    return "; ".join(route_strings)

def format_loads(loads):
    """Format the loads dictionary into a string representation."""
    if not loads:
        return "-"

    load_strings = []
    for vehicle, load in loads.items():
        load_strings.append(f"{vehicle}: {load}")

    return "; ".join(load_strings)


def make_text_table(results, columns):
    """Create a fixed-width text table with separate columns."""
    headers = [title for _, title in columns]
    rows = [
        [str(result[key]) if result[key] not in ("", None) else "-" for key, _ in columns]
        for result in results
    ]

    widths = [
        max(len(headers[i]), max(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def format_row(row):
        cells = [row[i].ljust(widths[i]) for i in range(len(row))]
        return "| " + " | ".join(cells) + " |"

    lines = [separator, format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    lines.append(separator)
    return "\n".join(lines)


def save_report(report_str, output_directory, file_name):
    """Save one scenario report and return its path."""
    report_file = Path(output_directory) / file_name
    report_file.write_text(report_str, encoding="utf-8")
    return report_file


def save_text_report(results, columns, title, output_file):
    """Save results in a simple presentation-ready text report."""
    report_lines = [title, "=" * len(title), "", make_text_table(results, columns)]

    # Add route details if available
    if "routes" in results[0]:
        report_lines.extend(["", "ROUTE DETAILS", "=" * 13])

        for result in results:
            if "demand_profile" in result:
                label = f"{result['demand_profile']} (Q={result['capacity']})"
            elif "capacity" in result:
                label = f"Q={result['capacity']}"
            elif "case" in result:
                label = result["case"]
            else:
                label = result.get("scenario", "solution")

            report_lines.append(f"\nCase: {label}")
            report_lines.append(f"Routes: {format_routes(result['routes']) or '-'}")
            report_lines.append(f"Loads:  {format_loads(result['loads']) or '-'}")

            if "schedules" in result:
                report_lines.append(f"Times:  {result['schedules'] or '-'}")

    report_str = "\n".join(report_lines) + "\n"
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(report_str, encoding="utf-8")
    return output_file

def append_text_report(results, columns, title, output_file):
    """Append results to an existing text report."""
    report_lines = [title, "=" * len(title), "", make_text_table(results, columns)]

    if "routes" in results[0]:
        report_lines.extend(["", "ROUTE DETAILS", "=" * 13])

        for result in results:
            if "demand_profile" in result:
                label = f"{result['demand_profile']} (Q={result['capacity']})"
            elif "capacity" in result:
                label = f"Q={result['capacity']}"
            elif "case" in result:
                label = result["case"]
            else:
                label = result.get("scenario", "solution")

            report_lines.append(f"\nCase: {label}")
            report_lines.append(f"Routes: {format_routes(result['routes']) or '-'}")
            report_lines.append(f"Loads:  {format_loads(result['loads']) or '-'}")

            if "schedules" in result:
                report_lines.append(f"Times:  {result['schedules'] or '-'}")

    report_str = "\n".join(report_lines) + "\n"
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a", encoding="utf-8") as file:
        file.write(report_str)
    return output_file

def save_solver_result(result, variables, output_file, schedule_rows=None):
    """Save the selected solver decisions and routes for one case."""
    title = f"SOLVER RESULT - {result['case'].upper()}"
    report_lines = [title, "=" * len(title), ""]

    summary_fields = [
        ("status", "Status"),
        ("available_fleet", "Available fleet"),
        ("selected_fleet", "Selected fleet"),
        ("objective", "Objective"),
        ("active_vehicles", "Active vehicles"),
        ("active_capacity", "Active capacity"),
        ("capacity_lower_bound", "Capacity lower bound"),
        ("utilization_percent", "Utilization (%)"),
        ("total_distance", "Total distance"),
        ("total_travel_time", "Total travel time"),
        ("maximum_route_time", "Maximum route time"),
        ("fixed_cost", "Fixed cost"),
        ("distance_cost", "Distance cost"),
        ("time_cost", "Time cost"),
        ("variable_cost", "Variable cost"),
        ("solver_termination", "Solver termination"),
        ("best_bound", "Best bound"),
        ("relative_gap_percent", "Relative gap (%)"),
        ("runtime_seconds", "Runtime (s)"),
    ]

    for key, label in summary_fields:
        if key in result:
            value = result[key]
            report_lines.append(f"{label}: {value if value not in ('', None) else '-'}")

    report_lines.extend(["", "ROUTES", "------"])
    report_lines.extend(result.get("routes", "").split("; ") if result.get("routes") else ["-"])
    report_lines.extend(["", "LOADS", "-----"])
    report_lines.extend(result.get("loads", "").split("; ") if result.get("loads") else ["-"])

    if result.get("depot_details"):
        report_lines.extend(["", "DEPOT DETAILS", "-------------"])
        report_lines.extend(result["depot_details"].split("; "))

    if result["status"].startswith(("Optimal", "Feasible")):
        active = [
            f"z[{vehicle}] = 1"
            for vehicle in sorted(variables["z"])
            if pulp.value(variables["z"][vehicle]) > 0.5
        ]
        assignments = [
            f"y[{customer},{vehicle}] = 1"
            for customer, vehicle in sorted(variables["y"])
            if pulp.value(variables["y"][customer, vehicle]) > 0.5
        ]
        arcs = [
            f"x[{first},{second},{vehicle}] = 1"
            for first, second, vehicle in sorted(variables["x"])
            if pulp.value(variables["x"][first, second, vehicle]) > 0.5
        ]

        report_lines.extend(["", "ACTIVE VEHICLES", "---------------", *active])
        report_lines.extend(["", "CUSTOMER ASSIGNMENTS", "--------------------", *assignments])
        report_lines.extend(["", "SELECTED ARCS", "-------------", *arcs])

        if schedule_rows:
            schedule_results = [
                {
                    "vehicle": row["vehicle"],
                    "customer": row["customer"],
                    "window": f"[{row['window_start']},{row['window_end']}]",
                    "arrival": f"{row['arrival']:.2f}",
                    "service_start": f"{row['service_start']:.2f}",
                    "waiting": f"{row['waiting']:.2f}",
                }
                for row in schedule_rows
            ]
            schedule_columns = [
                ("vehicle", "Vehicle"),
                ("customer", "Customer"),
                ("window", "Window"),
                ("arrival", "Arrival"),
                ("service_start", "Service t"),
                ("waiting", "Waiting"),
            ]
            report_lines.extend(
                [
                    "",
                    "EARLIEST VALID SERVICE SCHEDULE",
                    "-------------------------------",
                    make_text_table(schedule_results, schedule_columns),
                ]
            )

    report_str = "\n".join(report_lines) + "\n"
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(report_str, encoding="utf-8")
    return output_file


def lp_relaxation(scenario, vehicle_types, capacity):
    """Build and solve a fresh LP relaxation for one capacity value."""
    from model import build_model

    scenario = deepcopy(scenario)
    vehicle_types = deepcopy(vehicle_types)
    vehicle_types["van"]["capacity"] = capacity

    model, variables = build_model(scenario, vehicle_types)

    # Only the binary decision variables are relaxed.
    for group in ("x", "y", "z"):
        for variable in variables[group].values():
            variable.cat = pulp.LpContinuous

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[model.status]

    objective = None
    sum_z = None
    fractional_counts = {"x": 0, "y": 0, "z": 0}
    fractional_examples = {"x": [], "y": [], "z": []}

    if status == "Optimal":
        objective = round(pulp.value(model.objective), 6)
        sum_z = round(
            sum(pulp.value(variable) or 0.0 for variable in variables["z"].values()),
            6,
        )

        for group in ("x", "y", "z"):
            for index, variable in variables[group].items():
                value = pulp.value(variable)
                if value is None or not 1e-6 < value < 1 - 1e-6:
                    continue

                fractional_counts[group] += 1
                if len(fractional_examples[group]) < 3:
                    if not isinstance(index, tuple):
                        index = (index,)
                    label = ",".join(str(part) for part in index)
                    fractional_examples[group].append(
                        f"{group}[{label}]={value:.4f}"
                    )

    example_text = []
    for group in ("y", "z", "x"):
        example_text.extend(fractional_examples[group])

    return {
        "status": status,
        "objective": objective,
        "capacity": capacity,
        "sum_z": sum_z,
        "fractional_x": fractional_counts["x"],
        "fractional_y": fractional_counts["y"],
        "fractional_z": fractional_counts["z"],
        "fractional_examples": "; ".join(example_text) or "-",
    }
