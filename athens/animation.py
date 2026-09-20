from bisect import bisect_right
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import osmnx as ox

from .helpers import get_shortest_parallel_edge
from .visualization import (
    LABEL_POSITIONS,
    edge_coordinates,
    node_position,
    route_coordinates,
    vehicle_colors,
)


def timed_road_points(graph, road_nodes, vehicle_factor, duration):
    """Return road coordinates with elapsed travel time at each point."""
    coordinates = [node_position(graph, road_nodes[0])]
    elapsed_times = [0.0]
    elapsed = 0.0

    for first, second in zip(road_nodes[:-1], road_nodes[1:]):
        points = edge_coordinates(graph, first, second)
        edge = get_shortest_parallel_edge(graph, first, second)
        edge_time = float(edge["travel_time"]) * vehicle_factor / 60
        lengths = [
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(points[:-1], points[1:])
        ]
        if not lengths:
            continue
        total_length = sum(lengths)

        for point, length in zip(points[1:], lengths):
            share = length / total_length if total_length else 1 / len(lengths)
            elapsed += edge_time * share
            coordinates.append(point)
            elapsed_times.append(elapsed)

    if elapsed_times[-1] > 0:
        scale = duration / elapsed_times[-1]
        elapsed_times = [value * scale for value in elapsed_times]

    return coordinates, elapsed_times


def point_at_time(coordinates, elapsed_times, elapsed):
    """Interpolate one position along a timed polyline."""
    if elapsed <= 0:
        return coordinates[0]
    if elapsed >= elapsed_times[-1]:
        return coordinates[-1]

    index = bisect_right(elapsed_times, elapsed)
    start_time = elapsed_times[index - 1]
    end_time = elapsed_times[index]
    progress = (elapsed - start_time) / (end_time - start_time)
    start = coordinates[index - 1]
    end = coordinates[index]
    return (
        start[0] + progress * (end[0] - start[0]),
        start[1] + progress * (end[1] - start[1]),
    )


def build_animation_segments(
    scenario,
    vehicle_types,
    road_routes,
    graph=None,
):
    """Build travel, waiting, and service segments on one common clock."""
    timelines = {}

    for vehicle, route_data in road_routes.items():
        vehicle_type = vehicle.split("_")[0]
        factor = vehicle_types[vehicle_type]["travel_time_factor"]
        services = {
            service["customer"]: service for service in route_data["services"]
        }
        segments = []

        for hop in route_data["hops"]:
            duration = hop["arrival"] - hop["departure"]
            if graph is None:
                coordinates = [
                    _milp_position(scenario, hop["origin"]),
                    _milp_position(scenario, hop["destination"]),
                ]
                elapsed_times = [0.0, duration]
            else:
                coordinates, elapsed_times = timed_road_points(
                    graph,
                    hop["road_nodes"],
                    factor,
                    duration,
                )

            segments.append(
                {
                    "kind": "travel",
                    "start": hop["departure"],
                    "end": hop["arrival"],
                    "coordinates": coordinates,
                    "elapsed_times": elapsed_times,
                    "label": f"travelling {hop['origin']} -> {hop['destination']}",
                }
            )

            if hop["destination"] not in services:
                continue

            service = services[hop["destination"]]
            customer_position = coordinates[-1]

            if service["waiting"] > 0:
                segments.append(
                    {
                        "kind": "waiting",
                        "start": service["arrival"],
                        "end": service["service_start"],
                        "coordinates": [customer_position],
                        "elapsed_times": [0.0],
                        "label": f"waiting at C{service['customer']}",
                    }
                )

            segments.append(
                {
                    "kind": "service",
                    "start": service["service_start"],
                    "end": service["service_end"],
                    "coordinates": [customer_position],
                    "elapsed_times": [0.0],
                    "label": f"servicing C{service['customer']}",
                }
            )

        timelines[vehicle] = segments

    return timelines


def _milp_position(scenario, node):
    data = scenario["customers"].get(node, scenario["depots"].get(node))
    return data["coordinates"]


def _vehicle_state(timeline, current_time):
    first_position = timeline[0]["coordinates"][0]

    for segment in timeline:
        if segment["start"] <= current_time < segment["end"]:
            if segment["kind"] == "travel":
                position = point_at_time(
                    segment["coordinates"],
                    segment["elapsed_times"],
                    current_time - segment["start"],
                )
            else:
                position = segment["coordinates"][0]
            return position, segment["kind"], segment["label"]

    if current_time < timeline[0]["start"]:
        return first_position, "waiting", "waiting to depart"
    return timeline[-1]["coordinates"][-1], "returned", "returned to depot"


def _service_state(service, current_time):
    if current_time < service["service_start"]:
        return "white"
    if current_time < service["service_end"]:
        return "gold"
    return "lightgreen"


def _draw_locations(
    ax,
    scenario,
    service_lookup,
    position_function,
):
    customer_markers = {}

    for depot in scenario["depots"]:
        x, y = position_function(depot)
        ax.scatter(x, y, marker="s", s=105, color="orange", edgecolors="black", zorder=7)
        ax.annotate(depot, (x, y), xytext=(7, 7), textcoords="offset points", fontsize=9)

    for customer, data in scenario["customers"].items():
        x, y = position_function(customer)
        dx, dy, horizontal, vertical = LABEL_POSITIONS[customer]
        service_start = service_lookup[customer]["service_start"]
        marker = ax.scatter(
            x,
            y,
            s=58,
            color="white",
            edgecolors="black",
            zorder=7,
        )
        customer_markers[customer] = marker
        ax.annotate(
            f"C{customer}\nt={service_start:.1f}",
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            ha=horizontal,
            va=vertical,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1),
            zorder=8,
        )

    return customer_markers


def save_route_animation(
    scenario,
    vehicle_types,
    road_routes,
    loads,
    output_file,
    graph=None,
    location_nodes=None,
    minutes_per_frame=1.0,
    frames_per_second=4,
):
    """Save a slow, simultaneous animation on the MILP or road graph."""
    vehicles = sorted(road_routes)
    colors = vehicle_colors(vehicles)
    timelines = build_animation_segments(
        scenario,
        vehicle_types,
        road_routes,
        graph,
    )
    service_lookup = {
        service["customer"]: service
        for route_data in road_routes.values()
        for service in route_data["services"]
    }

    if graph is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        position_function = lambda node: _milp_position(scenario, node)
        for vehicle in vehicles:
            coordinates = [
                position_function(node)
                for node in road_routes[vehicle]["milp_route"]
            ]
            ax.plot(
                [point[0] for point in coordinates],
                [point[1] for point in coordinates],
                color=colors[vehicle],
                linewidth=2,
                alpha=0.35,
            )
        title = "MILP graph animation"
    else:
        fig, ax = ox.plot_graph(
            graph,
            node_size=0,
            edge_color="lightgray",
            edge_linewidth=0.5,
            bgcolor="white",
            figsize=(12, 8),
            show=False,
            close=False,
        )
        position_function = lambda node: node_position(graph, location_nodes[node])
        all_coordinates = []
        for vehicle in vehicles:
            coordinates = route_coordinates(graph, road_routes[vehicle]["road_nodes"])
            all_coordinates.extend(coordinates)
            ax.plot(
                [point[0] for point in coordinates],
                [point[1] for point in coordinates],
                color=colors[vehicle],
                linewidth=2.2,
                alpha=0.45,
            )
        xs = [point[0] for point in all_coordinates]
        ys = [point[1] for point in all_coordinates]
        x_margin = (max(xs) - min(xs)) * 0.08
        y_margin = (max(ys) - min(ys)) * 0.10
        ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
        ax.set_ylim(min(ys) - y_margin, max(ys) + y_margin)
        title = "Athens road-network animation"

    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    fig.subplots_adjust(right=0.76)
    customer_markers = _draw_locations(
        ax,
        scenario,
        service_lookup,
        position_function,
    )

    vehicle_markers = {}
    marker_shapes = {"motorcycle": "o", "van": "s", "truck": "D"}
    for vehicle in vehicles:
        vehicle_type = vehicle.split("_")[0]
        marker, = ax.plot(
            [],
            [],
            marker=marker_shapes[vehicle_type],
            markersize=11,
            color=colors[vehicle],
            markeredgecolor="black",
            linestyle="None",
            zorder=10,
        )
        vehicle_markers[vehicle] = marker

    clock_text = ax.text(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        fontsize=12,
        va="top",
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9),
        zorder=12,
    )
    fig.text(0.78, 0.84, "Vehicle status", fontsize=11)
    status_text = fig.text(
        0.78,
        0.80,
        "",
        fontsize=8.5,
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=2),
    )
    fig.text(
        0.78,
        0.44,
        "Customer colours\nwhite: pending\ngold: in service\ngreen: completed",
        fontsize=8.5,
        va="top",
    )

    route_legend = []
    for vehicle in vehicles:
        vehicle_type = vehicle.split("_")[0]
        capacity = vehicle_types[vehicle_type]["capacity"]
        route_legend.append(
            Line2D(
                [0],
                [0],
                color=colors[vehicle],
                linewidth=2.5,
                marker=marker_shapes[vehicle_type],
                label=f"{vehicle}  {loads[vehicle]}/{capacity}",
            )
        )
    fig.legend(handles=route_legend, loc="lower right", bbox_to_anchor=(0.985, 0.08), fontsize=8)

    final_time = max(route["finish_time"] for route in road_routes.values())
    frame_count = math.ceil(final_time / minutes_per_frame) + 1

    def update(frame):
        current_time = min(frame * minutes_per_frame, final_time)
        clock_text.set_text(f"t = {current_time:.0f} min")
        status_lines = []

        for vehicle in vehicles:
            position, _, status = _vehicle_state(timelines[vehicle], current_time)
            vehicle_markers[vehicle].set_data([position[0]], [position[1]])
            status_lines.extend([vehicle, f"  {status}", ""])

        for customer, service in service_lookup.items():
            customer_markers[customer].set_facecolor(
                _service_state(service, current_time)
            )

        status_text.set_text("\n".join(status_lines))
        return [
            clock_text,
            status_text,
            *vehicle_markers.values(),
            *customer_markers.values(),
        ]

    animation = FuncAnimation(
        fig,
        update,
        frames=frame_count,
        interval=1000 / frames_per_second,
        blit=False,
        repeat=True,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.25)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.unlink(missing_ok=True)
    animation.save(
        output_file,
        writer=PillowWriter(fps=frames_per_second),
        dpi=90,
        savefig_kwargs={"facecolor": "white"},
    )
    plt.close(fig)
    print(f"Animation saved in: {output_file}")
    return final_time, frame_count
