import time

import pulp

from model import build_model, get_d_ij, get_t_ijk
from tools import extract_route, save_solver_result


def build_schedules(scenario, vehicle_types, routes):
    """Create the earliest feasible timetable for every selected route."""
    schedules = {}

    for vehicle, route in routes.items():
        vehicle_type = vehicle.split("_")[0]
        current_time = float(vehicle_types[vehicle_type]["start_time"])
        hops = [] 
        services = []

        for origin, destination in zip(route[:-1], route[1:]):
            departure = current_time # departure from the previous node 
            travel_time = get_t_ijk(scenario, vehicle_types, origin, destination, vehicle,)
            arrival = departure + travel_time # arrival to the current node 

            hops.append({"origin": origin, "destination": destination, "departure": departure, "arrival": arrival,})

            current_time = arrival

            if destination in scenario["customers"]:
                window_start, window_end = scenario["time_windows"][destination]
                # if gets there early, waits for the window to open 
                service_start = max(arrival, window_start)
                service_end = (service_start + scenario["customers"][destination]["service_time"])
                services.append(
                    {
                        "vehicle": vehicle,
                        "customer": destination,
                        "window_start": window_start,
                        "window_end": window_end,
                        "arrival": arrival,
                        "service_start": service_start,
                        "service_end": service_end,
                        "waiting": service_start - arrival,
                    }
                )
                current_time = service_end

        schedules[vehicle] = {
            "hops": hops,
            "services": services,
            "finish_time": current_time,
        }

    return schedules


def solve_athens_model(scenario, vehicle_types, output_file):
    """Solve the Athens MILP and return routes with their schedules."""

    # build the model 
    model, variables = build_model(scenario, vehicle_types)

    # solve the model 
    started = time.perf_counter()
    model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=120))
    runtime = time.perf_counter() - started
    status = pulp.LpStatus[model.status]

    print(f"Solver status: {status}")
    print(f"Runtime: {runtime:.2f} s")

    if status != "Optimal":
        return variables, {}, {}, None

    active_vehicles = sorted(vehicle for vehicle, variable in variables["z"].items() if pulp.value(variable) > 0.5)
    routes = {vehicle: extract_route(vehicle, variables["x"], len(scenario["customers"]),) for vehicle in active_vehicles}
    schedules = build_schedules(scenario, vehicle_types, routes)

    loads = {}
    total_distance = 0.0
    total_travel_time = 0.0
    fixed_cost = 0.0
    distance_cost = 0.0
    time_cost = 0.0

    for vehicle, route in routes.items():
        vehicle_type = vehicle.split("_")[0]
        loads[vehicle] = sum(
            scenario["customers"][customer]["demand"]
            for customer in route[1:-1]
        )

        for origin, destination in zip(route[:-1], route[1:]):
            distance = get_d_ij(scenario, origin, destination)
            travel_time = get_t_ijk(
                scenario,
                vehicle_types,
                origin,
                destination,
                vehicle,
            )
            total_distance += distance
            total_travel_time += travel_time
            distance_cost += distance * vehicle_types[vehicle_type]["cost_per_distance"]
            time_cost += travel_time * vehicle_types[vehicle_type]["cost_per_time"]

        fixed_cost += vehicle_types[vehicle_type]["fixed_cost"]

    active_capacity = sum(
        vehicle_types[vehicle.split("_")[0]]["capacity"]
        for vehicle in active_vehicles
    )
    total_demand = sum(
        customer["demand"] for customer in scenario["customers"].values()
    )
    route_text = "; ".join(
        f"{vehicle}: {' -> '.join(route)}"
        for vehicle, route in routes.items()
    )
    load_text = "; ".join(
        f"{vehicle}={loads[vehicle]}/{vehicle_types[vehicle.split('_')[0]]['capacity']}"
        for vehicle in active_vehicles
    )

    result = {
        "case": "Athens integrated scenario",
        "status": status,
        "selected_fleet": ", ".join(active_vehicles),
        "objective": round(pulp.value(model.objective), 3),
        "active_vehicles": len(active_vehicles),
        "active_capacity": active_capacity,
        "utilization_percent": round(100 * total_demand / active_capacity, 2),
        "total_distance": round(total_distance, 3),
        "total_travel_time": round(total_travel_time, 3),
        "maximum_route_time": round(max(schedule["finish_time"] for schedule in schedules.values()), 3,),
        "fixed_cost": round(fixed_cost, 3),
        "distance_cost": round(distance_cost, 3),
        "time_cost": round(time_cost, 3),
        "variable_cost": round(distance_cost + time_cost, 3),
        "runtime_seconds": round(runtime, 3),
        "routes": route_text,
        "loads": load_text,
    }

    schedule_rows = [
        service
        for vehicle in active_vehicles
        for service in schedules[vehicle]["services"]
    ]
    save_solver_result(result, variables, output_file, schedule_rows)

    print(f"Objective: {result['objective']}")
    for route in route_text.split("; "):
        print(route)

    return variables, routes, schedules, result
