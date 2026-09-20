from copy import deepcopy
import json
from pathlib import Path
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


def save_text_report(results, columns, title, output_file):
    """Save results in a simple presentation-ready text report."""
    report = [title, "=" * len(title), "", make_text_table(results, columns)]

    if "routes" in results[0]:
        report.extend(["", "ROUTE DETAILS", "=" * 13])

        for result in results:
            if "demand_profile" in result:
                label = f"{result['demand_profile']} (Q={result['capacity']})"
            elif "capacity" in result:
                label = f"Q={result['capacity']}"
            elif "case" in result:
                label = result["case"]
            else:
                label = result.get("scenario", "solution")

            report.append(f"\nCase: {label}")
            report.append(f"Routes: {format_routes(result['routes']) or '-'}")
            report.append(f"Loads:  {format_loads(result['loads']) or '-'}")

            if "schedules" in result:
                report.append(f"Times:  {result['schedules'] or '-'}")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(report) + "\n", encoding="utf-8")
    return output_file

def append_text_report(results, columns, title, output_file):
    """Append results to an existing text report."""
    report = [title, "=" * len(title), "", make_text_table(results, columns)]

    if "routes" in results[0]:
        report.extend(["", "ROUTE DETAILS", "=" * 13])

        for result in results:
            if "demand_profile" in result:
                label = f"{result['demand_profile']} (Q={result['capacity']})"
            elif "capacity" in result:
                label = f"Q={result['capacity']}"
            elif "case" in result:
                label = result["case"]
            else:
                label = result.get("scenario", "solution")

            report.append(f"\nCase: {label}")
            report.append(f"Routes: {format_routes(result['routes']) or '-'}")
            report.append(f"Loads:  {format_loads(result['loads']) or '-'}")

            if "schedules" in result:
                report.append(f"Times:  {result['schedules'] or '-'}")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a", encoding="utf-8") as file:
        file.write("\n".join(report) + "\n")
    return output_file

def save_solver_result(result, variables, output_file, schedule_rows=None):
    """Save the selected solver decisions and routes for one case."""
    title = f"SOLVER RESULT - {result['case'].upper()}"
    lines = [title, "=" * len(title), ""]

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
            lines.append(f"{label}: {value if value not in ('', None) else '-'}")

    lines.extend(["", "ROUTES", "------"])
    lines.extend(result.get("routes", "").split("; ") if result.get("routes") else ["-"])
    lines.extend(["", "LOADS", "-----"])
    lines.extend(result.get("loads", "").split("; ") if result.get("loads") else ["-"])

    if result.get("depot_details"):
        lines.extend(["", "DEPOT DETAILS", "-------------"])
        lines.extend(result["depot_details"].split("; "))

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

        lines.extend(["", "ACTIVE VEHICLES", "---------------", *active])
        lines.extend(["", "CUSTOMER ASSIGNMENTS", "--------------------", *assignments])
        lines.extend(["", "SELECTED ARCS", "-------------", *arcs])

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
            lines.extend(
                [
                    "",
                    "EARLIEST VALID SERVICE SCHEDULE",
                    "-------------------------------",
                    make_text_table(schedule_results, schedule_columns),
                ]
            )

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
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

