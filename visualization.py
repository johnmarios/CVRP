from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import pulp

from tools import extract_route


def plot_solution(
    scenario,
    variables,
    vehicle_types,
    output_file,
    show=True,
    show_time_windows=False,
    service_times=None,
    label_positions=None,
):
    """Plot the optimized directed vehicle routes."""
    fig, ax = plt.subplots(figsize=(9, 7))

    default_label_positions = {
        "1": (-10, 8, "right", "bottom"),
        "2": (0, 12, "center", "bottom"),
        "3": (10, 2, "left", "bottom"),
        "4": (-10, 7, "right", "bottom"),
        "5": (0, -12, "center", "top"),
        "6": (-10, 5, "right", "bottom"),
    }
    label_positions = {
        **default_label_positions,
        **(label_positions or {}),
    }
    depot_label_positions = {
        "D2": (-12, 8, "right", "bottom"),
    }

    for depot, data in scenario["depots"].items():
        x, y = data["coordinates"]
        dx, dy, horizontal, vertical = depot_label_positions.get(
            depot,
            (8, 7, "left", "bottom")
        )
        ax.scatter(x, y, marker="s", s=120, color="orange", edgecolors="black", zorder=5)
        ax.annotate(
            depot,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=11,
            ha=horizontal,
            va=vertical,
            zorder=6
        )

    for customer, data in scenario["customers"].items():
        x, y = data["coordinates"]
        dx, dy, horizontal, vertical = label_positions.get(
            customer,
            (8, 6, "left", "bottom")
        )
        ax.scatter(x, y, s=70, color="white", edgecolors="black", linewidths=1.2, zorder=5)
        if service_times is None:
            customer_label = f"C{customer}\nq={data['demand']}"
        else:
            customer_label = f"C{customer}  q={data['demand']}"

        if show_time_windows and customer in scenario.get("time_windows", {}):
            start, end = scenario["time_windows"][customer]
            customer_label += f"\nTW=[{start},{end}]"
        if service_times is not None and customer in service_times:
            customer_label += f"\nt={service_times[customer]:.1f}"

        ax.annotate(
            customer_label,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=10,
            ha=horizontal,
            va=vertical,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1),
            zorder=6
        )

    if "z" in variables:
        plotted_routes = (
            (
                vehicle,
                extract_route(vehicle, variables["x"], len(scenario["customers"])),
                sum(
                    scenario["customers"][customer]["demand"]
                    for customer in scenario["customers"]
                    if pulp.value(variables["y"][customer, vehicle]) > 0.5
                ),
            )
            for vehicle, active in variables["z"].items()
            if pulp.value(active) >= 0.5
        )
    else:
        plotted_routes = (
            (
                vehicle,
                route,
                sum(
                    scenario["customers"][customer]["demand"]
                    for customer in route
                    if customer in scenario["customers"]
                ),
            )
            for vehicle, route in variables.items()
        )

    for vehicle, route, load in plotted_routes:
        vehicle_type = vehicle.split("_")[0]
        capacity = vehicle_types[vehicle_type]["capacity"]

        route_coordinates = []
        for node in route:
            node_data = scenario["customers"].get(node, scenario["depots"].get(node))
            route_coordinates.append(node_data["coordinates"])

        xs = [point[0] for point in route_coordinates]
        ys = [point[1] for point in route_coordinates]
        line, = ax.plot(
            xs,
            ys,
            linewidth=2,
            zorder=2,
            label=f"{vehicle} | load {load}/{capacity}"
        )

        for i in range(len(xs) - 1):
            ax.annotate(
                "",
                xy=(xs[i + 1], ys[i + 1]),
                xytext=(xs[i], ys[i]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=line.get_color(),
                    linewidth=1.8,
                    mutation_scale=14,
                    shrinkA=8,
                    shrinkB=8
                ),
                zorder=3
            )

    ax.set_title(f"VRP Solution - {scenario['name']}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.margins(0.22 if show_time_windows else 0.12)
    ax.grid()
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    fig.tight_layout()

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.unlink(missing_ok=True)
    with output_file.open("wb") as image_file:
        fig.savefig(image_file, format="png", dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def plot_capacity_results(results, capacities, output_file, show=False):
    """Plot objective value and required vehicles for each capacity."""
    optimal = [result for result in results if result["status"] == "Optimal"]
    fig, (ax_cost, ax_vehicles) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax_cost.plot(
        [result["capacity"] for result in optimal],
        [result["objective"] for result in optimal],
        marker="o"
    )
    ax_cost.set_ylabel("Total cost")
    ax_cost.set_title("Capacity sensitivity")
    ax_cost.grid(alpha=0.3)

    for result in results:
        if result["status"] != "Optimal":
            ax_cost.text(
                result["capacity"],
                0.05,
                (
                    "Infeasible\n"
                    f"LB={result['capacity_lower_bound']} > "
                    f"{result['available_vehicles']}"
                ),
                color="red",
                ha="center",
                va="bottom",
                transform=ax_cost.get_xaxis_transform()
            )

    first_feasible = min(result["capacity"] for result in optimal)
    one_vehicle_cases = [
        result["capacity"]
        for result in optimal
        if result["active_vehicles"] == 1
    ]
    ax_cost.axvline(first_feasible, color="gray", linestyle=":", alpha=0.7)
    if one_vehicle_cases:
        ax_cost.axvline(min(one_vehicle_cases), color="gray", linestyle=":", alpha=0.7)

    ax_vehicles.step(
        [result["capacity"] for result in optimal],
        [result["active_vehicles"] for result in optimal],
        where="mid",
        marker="o",
        label="Active vehicles"
    )
    ax_vehicles.step(
        [result["capacity"] for result in results],
        [result["capacity_lower_bound"] for result in results],
        where="mid",
        linestyle="--",
        label="Capacity lower bound"
    )
    available_vehicles = results[0]["available_vehicles"]
    ax_vehicles.axhline(
        available_vehicles,
        color="gray",
        linestyle=":",
        label=f"Available vehicles = {available_vehicles}",
    )
    ax_vehicles.set_xlabel("Vehicle capacity Q")
    ax_vehicles.set_ylabel("Vehicles")
    ax_vehicles.set_xticks(capacities)
    ax_vehicles.grid(alpha=0.3)
    ax_vehicles.legend()

    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def plot_scenario1_comparison(
    demand_profiles,
    capacity,
    relaxation_results,
    output_file,
    show=False,
):
    """Plot demand concentration and the MILP-LP comparison."""
    fig, (ax_demands, ax_relaxation) = plt.subplots(1, 2, figsize=(12, 4.8))

    customers = list(range(1, len(next(iter(demand_profiles.values()))) + 1))
    width = 0.36
    profile_names = list(demand_profiles)

    for index, profile_name in enumerate(profile_names):
        offset = (index - (len(profile_names) - 1) / 2) * width
        ax_demands.bar(
            [customer + offset for customer in customers],
            demand_profiles[profile_name],
            width,
            label=profile_name,
        )

    ax_demands.axhline(
        capacity / 2,
        color="red",
        linestyle="--",
        label=f"Q/2 = {capacity / 2:g}",
    )
    ax_demands.set_title("Demand distribution at Q=11")
    ax_demands.set_xlabel("Customer")
    ax_demands.set_ylabel("Demand")
    ax_demands.set_xticks(customers)
    ax_demands.set_xticklabels([f"C{customer}" for customer in customers])
    ax_demands.grid(axis="y", alpha=0.3)
    ax_demands.legend()

    positions = list(range(len(relaxation_results)))
    milp_values = [
        result["milp_objective"] or 0
        for result in relaxation_results
    ]
    lp_values = [
        result["lp_objective"] or 0
        for result in relaxation_results
    ]
    bars_milp = ax_relaxation.bar(
        [position - width / 2 for position in positions],
        milp_values,
        width,
        label="MILP",
    )
    bars_lp = ax_relaxation.bar(
        [position + width / 2 for position in positions],
        lp_values,
        width,
        label="LP relaxation",
    )

    for bars, values in ((bars_milp, milp_values), (bars_lp, lp_values)):
        for bar, value in zip(bars, values):
            if value:
                ax_relaxation.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    maximum_value = max(milp_values + lp_values)
    for position, result in zip(positions, relaxation_results):
        if result["milp_status"] != "Optimal":
            ax_relaxation.text(
                position - width / 2,
                maximum_value * 0.03,
                "Infeasible",
                color="red",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8,
            )

    ax_relaxation.set_title("MILP and LP relaxation")
    ax_relaxation.set_ylabel("Objective value")
    ax_relaxation.set_xticks(positions)
    ax_relaxation.set_xticklabels(
        [result["short_name"] for result in relaxation_results],
        rotation=15,
        ha="right",
    )
    ax_relaxation.grid(axis="y", alpha=0.3)
    ax_relaxation.legend()

    fig.tight_layout()
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def plot_relaxation_results(results, output_file):
    """Compare MILP objectives with their LP lower bounds."""
    positions = list(range(len(results)))
    width = 0.36

    milp_values = [result["milp_objective"] or 0 for result in results]
    lp_values = [result["lp_objective"] or 0 for result in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    milp_bars = ax.bar(
        [position - width / 2 for position in positions],
        milp_values,
        width,
        label="MILP"
    )
    lp_bars = ax.bar(
        [position + width / 2 for position in positions],
        lp_values,
        width,
        label="LP relaxation"
    )

    for bars, values in ((milp_bars, milp_values), (lp_bars, lp_values)):
        for bar, value in zip(bars, values):
            if value:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9
                )

    maximum = max(lp_values + milp_values)
    for position, result in zip(positions, results):
        if result["milp_status"] != "Optimal":
            ax.text(
                position - width / 2,
                maximum * 0.04,
                "MILP infeasible",
                ha="center",
                va="bottom",
                rotation=90,
                color="red",
                fontsize=9
            )

    ax.set_title("MILP and LP relaxation")
    ax.set_ylabel("Objective value")
    ax.set_xticks(positions)
    ax.set_xticklabels([result["case"].replace(" ", "\n", 1) for result in results])
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fleet_results(results, output_file):
    """Compare fixed and variable cost for the fleet cases."""
    labels = [result["case"] for result in results]
    fixed_costs = [result["fixed_cost"] or 0 for result in results]
    variable_costs = [result["variable_cost"] or 0 for result in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    fixed_bars = ax.bar(labels, fixed_costs, label="Fixed cost")
    variable_bars = ax.bar(
        labels,
        variable_costs,
        bottom=fixed_costs,
        label="Variable cost"
    )

    for position, (fixed_cost, variable_cost) in enumerate(
        zip(fixed_costs, variable_costs)
    ):
        total_cost = fixed_cost + variable_cost
        if total_cost:
            ax.text(
                position,
                total_cost,
                f"{total_cost:.2f}",
                ha="center",
                va="bottom"
            )

    ax.set_title("Fleet composition cost comparison")
    ax.set_ylabel("Total cost")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=[fixed_bars, variable_bars])

    fig.tight_layout()
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_truck_cost_relaxation(
    fixed_cost_results,
    distance_cost_results,
    fixed_cost_break_even,
    distance_cost_break_even,
    output_file
):
    """Show when one truck becomes cheaper than two vans."""
    fig, (ax_fixed, ax_distance) = plt.subplots(1, 2, figsize=(12, 5))

    fixed_rows = sorted(fixed_cost_results, key=lambda row: row["truck_fixed_cost"])
    van_cost = fixed_rows[0]["two_vans_cost"]
    ax_fixed.plot(
        [row["truck_fixed_cost"] for row in fixed_rows],
        [row["truck_total_cost"] for row in fixed_rows],
        marker="o",
        label="One truck"
    )
    ax_fixed.axhline(van_cost, color="tab:orange", linestyle="--", label="Two vans")
    ax_fixed.axvline(fixed_cost_break_even, color="black", linestyle=":")
    ax_fixed.set_title("Relaxation of truck fixed cost")
    ax_fixed.set_xlabel("Truck fixed cost")
    ax_fixed.set_ylabel("Total cost")
    ax_fixed.grid(alpha=0.3)
    ax_fixed.legend()

    distance_rows = sorted(
        distance_cost_results,
        key=lambda row: row["truck_cost_per_distance"]
    )
    ax_distance.plot(
        [row["truck_cost_per_distance"] for row in distance_rows],
        [row["truck_total_cost"] for row in distance_rows],
        marker="o",
        label="One truck"
    )
    ax_distance.axhline(
        van_cost,
        color="tab:orange",
        linestyle="--",
        label="Two vans"
    )
    ax_distance.axvline(distance_cost_break_even, color="black", linestyle=":")
    ax_distance.set_title("Relaxation of truck distance cost")
    ax_distance.set_xlabel("Truck cost per distance unit")
    ax_distance.set_ylabel("Total cost")
    ax_distance.grid(alpha=0.3)
    ax_distance.legend()

    fig.suptitle("Break-even analysis: one truck versus two vans")
    fig.tight_layout()
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_time_window_results(results, output_file):
    """Compare objective value and active vehicles across time-window cases."""
    positions = list(range(len(results)))
    labels = [result["short_name"] for result in results]
    fig, (ax_cost, ax_vehicles) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    optimal = [result for result in results if result["status"] == "Optimal"]
    optimal_positions = [results.index(result) for result in optimal]

    ax_cost.bar(
        optimal_positions,
        [result["objective"] for result in optimal],
        color="tab:blue"
    )
    for position, result in zip(optimal_positions, optimal):
        ax_cost.text(
            position,
            result["objective"],
            f"{result['objective']:.2f}",
            ha="center",
            va="bottom"
        )

    for position, result in zip(positions, results):
        if result["status"] != "Optimal":
            ax_cost.text(
                position,
                0.05,
                "Infeasible",
                color="red",
                ha="center",
                rotation=90,
                transform=ax_cost.get_xaxis_transform()
            )

    ax_cost.set_title("Effect of time windows")
    ax_cost.set_ylabel("Objective")
    ax_cost.grid(axis="y", alpha=0.3)

    ax_vehicles.plot(
        optimal_positions,
        [result["active_vehicles"] for result in optimal],
        marker="o",
        label="Active vehicles"
    )
    ax_vehicles.plot(
        positions,
        [result["capacity_lower_bound"] for result in results],
        marker="o",
        linestyle="--",
        label="Capacity lower bound"
    )
    ax_vehicles.plot(
        positions,
        [result["temporal_lower_bound"] for result in results],
        marker="o",
        linestyle=":",
        label="Temporal lower bound"
    )
    ax_vehicles.set_ylabel("Vehicles")
    ax_vehicles.set_xticks(positions)
    ax_vehicles.set_xticklabels(labels)
    ax_vehicles.set_ylim(bottom=0)
    ax_vehicles.grid(alpha=0.3)
    ax_vehicles.legend()

    fig.tight_layout()
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_time_window_schedule(schedule_rows, output_file):
    """Plot service starts inside their allowed time windows."""
    vehicles = sorted({row["vehicle"] for row in schedule_rows})
    vehicle_positions = {
        vehicle: position for position, vehicle in enumerate(vehicles)
    }

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for row in schedule_rows:
        y_position = vehicle_positions[row["vehicle"]]
        ax.plot(
            [row["window_start"], row["window_end"]],
            [y_position, y_position],
            color="lightgray",
            linewidth=10,
            solid_capstyle="butt"
        )
        ax.scatter(row["service_start"], y_position, color="tab:blue", zorder=3)
        ax.annotate(
            f"C{row['customer']}\n{row['service_start']:.1f}",
            (row["service_start"], y_position),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=9
        )

    ax.set_title("Tight feasible schedule")
    ax.set_xlabel("Time")
    ax.set_yticks(range(len(vehicles)))
    ax.set_yticklabels(vehicles)
    ax.set_ylim(-0.4, len(vehicles) - 0.4 + 0.8)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_time_window_animation(
    scenario,
    variables,
    vehicle_types,
    schedule_rows,
    output_file,
    minutes_per_frame=1.0,
    frames_per_second=10
):
    """Animate all active vehicles on the same time axis."""
    active_vehicles = sorted(
        vehicle
        for vehicle, variable in variables["z"].items()
        if pulp.value(variable) > 0.5
    )

    def coordinates(node):
        data = scenario["customers"].get(node, scenario["depots"].get(node))
        return data["coordinates"]

    schedules = {
        vehicle: [row for row in schedule_rows if row["vehicle"] == vehicle]
        for vehicle in active_vehicles
    }
    routes = {
        vehicle: extract_route(
            vehicle,
            variables["x"],
            len(scenario["customers"])
        )
        for vehicle in active_vehicles
    }

    timelines = {}
    final_time = 0.0

    for vehicle in active_vehicles:
        vehicle_type = vehicle.split("_")[0]
        current_time = vehicle_types[vehicle_type]["start_time"]
        current_node = routes[vehicle][0]
        segments = []

        for row in schedules[vehicle]:
            customer = row["customer"]
            segments.append(
                {
                    "kind": "travel",
                    "start": current_time,
                    "end": row["arrival"],
                    "start_position": coordinates(current_node),
                    "end_position": coordinates(customer),
                    "label": f"travelling {current_node} -> C{customer}",
                }
            )

            if row["service_start"] > row["arrival"]:
                segments.append(
                    {
                        "kind": "waiting",
                        "start": row["arrival"],
                        "end": row["service_start"],
                        "start_position": coordinates(customer),
                        "end_position": coordinates(customer),
                        "label": f"waiting at C{customer}",
                    }
                )

            service_end = (
                row["service_start"]
                + scenario["customers"][customer]["service_time"]
            )
            segments.append(
                {
                    "kind": "service",
                    "start": row["service_start"],
                    "end": service_end,
                    "start_position": coordinates(customer),
                    "end_position": coordinates(customer),
                    "label": f"servicing C{customer}",
                }
            )
            current_time = service_end
            current_node = customer

        depot = routes[vehicle][-1]
        start_position = coordinates(current_node)
        end_position = coordinates(depot)
        distance = math.hypot(
            start_position[0] - end_position[0],
            start_position[1] - end_position[1]
        )
        return_time = (
            current_time
            + distance * scenario.get("minutes_per_distance_unit", 5.0)
        )
        segments.append(
            {
                "kind": "travel",
                "start": current_time,
                "end": return_time,
                "start_position": start_position,
                "end_position": end_position,
                "label": f"returning C{current_node} -> {depot}",
            }
        )

        timelines[vehicle] = segments
        final_time = max(final_time, return_time)

    def vehicle_state(vehicle, current_time):
        for segment in timelines[vehicle]:
            if segment["start"] <= current_time < segment["end"]:
                start_x, start_y = segment["start_position"]
                end_x, end_y = segment["end_position"]
                duration = segment["end"] - segment["start"]
                progress = 0 if duration == 0 else (
                    current_time - segment["start"]
                ) / duration
                position = (
                    start_x + progress * (end_x - start_x),
                    start_y + progress * (end_y - start_y),
                )
                return position, segment["kind"], segment["label"]

        depot = routes[vehicle][-1]
        return coordinates(depot), "returned", f"returned to {depot}"

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.subplots_adjust(right=0.76)
    colors = plt.get_cmap("tab10")

    for depot, data in scenario["depots"].items():
        x, y = data["coordinates"]
        ax.scatter(
            x,
            y,
            marker="s",
            s=130,
            color="orange",
            edgecolors="black",
            zorder=5
        )
        ax.annotate(depot, (x, y), xytext=(8, 7), textcoords="offset points")

    for customer, data in scenario["customers"].items():
        x, y = data["coordinates"]
        window = scenario.get("time_windows", {}).get(
            customer,
            [0, scenario["horizon"]]
        )
        ax.scatter(
            x,
            y,
            s=70,
            color="white",
            edgecolors="black",
            zorder=5
        )
        ax.annotate(
            f"C{customer}  q={data['demand']}\nTW=[{window[0]},{window[1]}]",
            (x, y),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1)
        )

    moving_markers = {}
    moving_labels = {}

    for index, vehicle in enumerate(active_vehicles):
        color = colors(index)
        route_coordinates = [coordinates(node) for node in routes[vehicle]]
        ax.plot(
            [position[0] for position in route_coordinates],
            [position[1] for position in route_coordinates],
            color=color,
            linewidth=2,
            alpha=0.35,
            label=vehicle
        )
        marker, = ax.plot(
            [],
            [],
            marker="o",
            markersize=11,
            color=color,
            markeredgecolor="black",
            linestyle="None",
            zorder=8
        )
        moving_markers[vehicle] = marker
        moving_labels[vehicle] = ax.text(
            0,
            0,
            f"V{index + 1}",
            fontsize=9,
            ha="left",
            va="bottom",
            zorder=9
        )

    clock_text = ax.text(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        fontsize=12,
        va="top",
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9)
    )
    fig.text(0.79, 0.82, "Vehicle status", fontsize=11)
    status_text = fig.text(0.79, 0.76, "", fontsize=9, va="top")

    marker_shapes = {
        "travel": "o",
        "waiting": "s",
        "service": "D",
        "returned": "X",
    }

    def update(frame):
        current_time = frame * minutes_per_frame
        clock_text.set_text(f"t = {current_time:.0f} min")
        status_lines = []

        for index, vehicle in enumerate(active_vehicles):
            position, kind, status = vehicle_state(vehicle, current_time)
            moving_markers[vehicle].set_data([position[0]], [position[1]])
            moving_markers[vehicle].set_marker(marker_shapes[kind])
            moving_labels[vehicle].set_position(
                (position[0] + 0.15, position[1] + 0.20 + 0.22 * index)
            )
            status_lines.append(f"V{index + 1}: {status}")

        status_text.set_text("\n\n".join(status_lines))
        return [
            clock_text,
            status_text,
            *moving_markers.values(),
            *moving_labels.values(),
        ]

    frame_count = math.ceil(final_time / minutes_per_frame) + 1
    animation = FuncAnimation(
        fig,
        update,
        frames=frame_count,
        interval=1000 / frames_per_second,
        blit=False,
        repeat=True
    )

    ax.set_title("Time-window VRP animation - tight feasible")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.margins(0.20)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_file, writer=PillowWriter(fps=frames_per_second), dpi=120)
    plt.close(fig)
    return final_time, frame_count


def plot_multiple_depot_results(results, output_file):
    """Compare cost and distance across the multiple-depot cases."""
    positions = list(range(len(results)))
    labels = [result["short_name"] for result in results]
    fixed_costs = [result["fixed_cost"] for result in results]
    variable_costs = [result["variable_cost"] for result in results]
    distances = [result["total_distance"] for result in results]

    fig, (ax_cost, ax_distance) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax_cost.bar(positions, fixed_costs, label="Fixed cost")
    ax_cost.bar(
        positions,
        variable_costs,
        bottom=fixed_costs,
        label="Variable cost"
    )
    for position, result in zip(positions, results):
        ax_cost.text(
            position,
            result["objective"],
            f"{result['objective']:.2f}",
            ha="center",
            va="bottom"
        )

    ax_cost.set_title("Effect of depot location and fleet allocation")
    ax_cost.set_ylabel("Total cost")
    ax_cost.grid(axis="y", alpha=0.3)
    ax_cost.legend()

    bars = ax_distance.bar(positions, distances, color="tab:green")
    for bar, distance in zip(bars, distances):
        ax_distance.text(
            bar.get_x() + bar.get_width() / 2,
            distance,
            f"{distance:.2f}",
            ha="center",
            va="bottom"
        )

    ax_distance.set_ylabel("Total distance")
    ax_distance.set_xticks(positions)
    ax_distance.set_xticklabels(labels)
    ax_distance.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)
