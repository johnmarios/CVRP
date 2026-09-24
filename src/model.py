"""MILP model builder for the fixed VRP scenarios.

- x[i,j,k]: vehicle k uses arc i -> j
- y[i,k]: customer i is served by vehicle k
- z[k]: vehicle k is activated
- u[i,k]: cumulative delivered load (also eliminates subtours)
- T[i,k]: service start time, only when time windows are enabled

"""

from __future__ import annotations
from tools import readjson
import math
from typing import Any

import pulp
import json 
import os 


# SETS

def get_C(scenario: dict):
    """Set of customers."""
    return set(scenario["customers"].keys())


def get_D(scenario: dict):
    """Set of depots."""
    return set(scenario["depots"].keys())

def get_T(scenario: dict):
    """Set of vehicle types."""
    return set(scenario["vehicle_info"].keys())

def get_n_t(scenario: dict, t: str):
    """Number of vehicles of type t."""
    return sum(scenario["vehicle_info"][t]["depots"].values())

def get_K_t(scenario: dict, t: str):
    """Set of different vehicles of type t."""
    vehicles = set()

    depots = scenario["vehicle_info"][t]["depots"]

    for depot, count in depots.items():
        for i in range(count):
            vehicles.add(f"{t}_{depot}_{i+1}")

    # format : {van_D0_1, van_D0_2, van_D1_1, van_D1_2, van_D1_3, van_D2_1, van_D2_2, van_D2_3}
    return vehicles 
    
def get_K(scenario: dict):
    """Set of all available vehicles."""
    vehicles = set()

    for t in scenario["vehicle_info"]:
        vehicles.update(get_K_t(scenario, t))

    return vehicles


def get_tau(k: str):
    """Vehicle type of vehicle k."""
    return k.split("_")[0]

def get_h(k: str):
    """Depot of vehicle k."""
    return k.split("_")[1] # str


def get_N_k(scenario: dict, k: str):
    """Set of service nodes available to vehicle k."""
    return get_C(scenario).union({get_h(k)})

def get_A_k(scenario: dict, k: str):
    """Set of arcs available to vehicle k."""
    nodes = get_N_k(scenario, k)
    return {(i, j) for i in nodes for j in nodes if i != j}


# CUSTOMER PARAMETERS

def get_q_i(scenario: dict, i: str):
    """Demand of customer i."""
    return scenario["customers"][i]["demand"]

def get_s_i(scenario: dict, i: str):
    """Service time of customer i."""
    return scenario["customers"][i]["service_time"]

def get_a_i(scenario: dict, i: str):
    """Earliest allowed service time of customer i."""
    windows = scenario.get("time_windows", {})

    if i not in windows:
        return 0.0

    return float(windows[i][0])


def get_b_i(scenario: dict, i: str):
    """Latest allowed service time of customer i."""
    windows = scenario.get("time_windows", {})

    if i not in windows:
        return float(scenario["horizon"])

    return float(windows[i][1])

# VEHICLE PARAMETERS

def get_Q_k(vehicle_types: dict, k: str):
    """Capacity of vehicle k."""
    vehicle_type = get_tau(k)
    return vehicle_types[vehicle_type]["capacity"]

def get_Q_t(vehicle_types: dict, t: str):
    """Capacity of vehicle type t."""
    return vehicle_types[t]["capacity"]

def get_F_k(vehicle_types: dict, k: str):
    """Fixed cost of vehicle k."""
    vehicle_type = get_tau(k)
    return vehicle_types[vehicle_type]["fixed_cost"]

def get_F_t(vehicle_types: dict, t: str):
    """Fixed cost of vehicle type t."""
    return vehicle_types[t]["fixed_cost"]

def get_c_t_d(vehicle_types: dict, t: str):
    """Cost per distance unit of vehicle type t."""
    return vehicle_types[t]["cost_per_distance"]

def get_c_t_t(vehicle_types: dict, t: str):
    """Cost per time unit of vehicle type t."""
    return vehicle_types[t]["cost_per_time"]

def get_time_factor_t(vehicle_types: dict, t: str):
    """Travel-time multiplier of vehicle type t."""
    return float(vehicle_types[t].get("travel_time_factor", 1.0))

def get_H_t(vehicle_types: dict, t: str):
    """Maximum route duration of vehicle type t."""
    return vehicle_types[t]["max_route_duration"]

def get_A_t_start(vehicle_types: dict, t: str):
    """Earliest departure time of vehicle type t."""
    return vehicle_types[t]["start_time"]

def get_B_t_return(vehicle_types: dict, t: str):
    """Latest return time of vehicle type t."""
    return vehicle_types[t]["end_time"]


# NETWORK PARAMETERS

# helper
def node_coordinates(scenario: dict, node: str):
    if node in scenario["customers"]:
        return scenario["customers"][node]["coordinates"]

    return scenario["depots"][node]["coordinates"]

# in athens network, distance is given in distance matrix 
# and not calculated from coordinates
def get_d_ij(scenario: dict, i: str, j: str):
    """Distance between nodes i and j."""
    matrix = scenario.get("distance_matrix")
    if matrix:
        return float(matrix[i][j])

    coord_i = node_coordinates(scenario, i)
    coord_j = node_coordinates(scenario, j)
    return math.hypot(coord_i[0] - coord_j[0], coord_i[1] - coord_j[1])


# in athens network, travel time is given in a matrix 
# and not calculated from distance
def get_t_ij(scenario: dict, i: str, j: str):
    """Base travel time between nodes i and j."""
    matrix = scenario.get("travel_time_matrix")
    if matrix:
        return float(matrix[i][j])

    distance = get_d_ij(scenario, i, j)
    minutes_per_unit = float(scenario.get("minutes_per_distance_unit", 5.0))
    return distance * minutes_per_unit

def get_t_ijk(scenario: dict, vehicle_types: dict, i: str, j: str, k: str):
    """Travel time between i and j for vehicle k."""
    base_time = get_t_ij(scenario, i, j)

    if not scenario.get("use_vehicle_travel_time_factors", False):
        return base_time

    vehicle_type = get_tau(k)
    time_factor = get_time_factor_t(vehicle_types, vehicle_type)
    return base_time * time_factor

def get_c_ijk(scenario: dict, vehicle_types: dict, i: str, j: str, k: str):
    """Travel cost between nodes i and j for vehicle k."""
    vehicle_type = get_tau(k)
    dij = get_d_ij(scenario, i, j)
    tijk = get_t_ijk(scenario, vehicle_types, i, j, k)
    ctd = get_c_t_d(vehicle_types, vehicle_type)
    ctt = get_c_t_t(vehicle_types, vehicle_type)
    return ctd * dij + ctt * tijk

# OBJECTIVE FUNCTION
def add_objective(model, scenario, vehicle_types, x, z):
    """Objective function: minimize total cost."""
    K = get_K(scenario)
    model += (pulp.lpSum(get_F_k(vehicle_types, k) * z[k] for k in K)
        + pulp.lpSum(get_c_ijk(scenario, vehicle_types, i, j, k) * x[i, j, k] for k in K for (i, j) in get_A_k(scenario, k)))

# CONSTRAINTS
def add_core_constraints(model, scenario, x, y, z):
    """Add core CVRP constraints to the model."""
    C = get_C(scenario)
    K = get_K(scenario)

    # each customer is served exactly once
    for i in C:
        model += pulp.lpSum(y[i, k] for k in K) == 1, f"serve_once_{i}"

    # flow conservation
    # a vehicle that serves a customer must enter and leave that customer exactly once
    for k in K:
        N_k = get_N_k(scenario, k)
        for i in C:
            # enters customer i exactly once
            model += pulp.lpSum(x[j, i, k] for j in N_k if j != i) == y[i, k], f"flow_in_{i}_{k}"
            # leaves customer i exactly once
            model += pulp.lpSum(x[i, j, k] for j in N_k if j != i) == y[i, k], f"flow_out_{i}_{k}"

    # flow conservation at depot 
    # an active vehicle leaves its depot once and returns once
    for k in K:
        depot = get_h(k)
        # the vehicle leaves the depot exactly once
        model += pulp.lpSum(x[depot, i, k] for i in C) == z[k], f"depot_depart_{k}"
        # the vehicle returns to the depot exactly once
        model += pulp.lpSum(x[i, depot, k] for i in C) == z[k], f"depot_return_{k}"

    # assignment of a customer to a vehicle implies vehicle activation
    for k in K:
        for i in C:
            model += y[i, k] <= z[k], f"activate_{i}_{k}"

def add_capacity_constraints(model, scenario, vehicle_types, y, z):
    """Add capacity constraints to the model."""
    C = get_C(scenario)
    K = get_K(scenario)

    # the total demand, served by vehicle k cannot exceed its capacity
    for k in K:
        Q_k = get_Q_k(vehicle_types, k)
        model += (pulp.lpSum(get_q_i(scenario, i) * y[i, k] for i in C) <= Q_k * z[k], f"capacity_{k}")

        # A single demand larger than Q_k can never be assigned to vehicle k.
        for i in C:
            if get_q_i(scenario, i) > Q_k:
                model += y[i, k] == 0, f"incompatible_{i}_{k}"

def add_subtour_constraints(model, scenario, vehicle_types, x, y, u):
    """Add subtour elimination constraints to the model."""
    C = get_C(scenario)
    K = get_K(scenario)

    # handle uik constraints
    for k in K:
        Q_k = get_Q_k(vehicle_types, k)
        for i in C:
            q_i = get_q_i(scenario, i)
            # lower bound on load
            model += u[i, k] >= q_i * y[i, k], f"load_lb_{i}_{k}"
            # upper bound on load
            model += u[i, k] <= Q_k * y[i, k], f"load_ub_{i}_{k}"

        for i in C:
            for j in C:
                if i == j:
                    continue
                q_j = get_q_i(scenario, j)
                M_jk = Q_k + q_j # define big M 
                # load propagation constraint
                model += (u[j, k] >= u[i, k] + q_j - M_jk * (1 - x[i, j, k]), f"load_propagation_{i}_{j}_{k}")


def add_time_window_constraints(model, scenario, vehicle_types, x, y, z, T_var, P, R):
    """Add time window constraints to the model."""
    C = get_C(scenario)
    K = get_K(scenario)


    # time constraint for the client i to be served by vehicle k between the desired time window [a_i, b_i]
    for i in C:
        a_i = get_a_i(scenario, i)
        b_i = get_b_i(scenario, i)
        for k in K:
            model += T_var[i, k] >= a_i * y[i, k], f"tw_lb_{i}_{k}"
            model += T_var[i, k] <= b_i * y[i, k], f"tw_ub_{i}_{k}"

    # time continuity constraints for the vehicle k to serve customer i and j in sequence
    for k in K:
        for i in C:
            s_i = get_s_i(scenario, i)
            b_i = get_b_i(scenario, i)
            for j in C:
                if i == j:
                    continue
                t_ijk = get_t_ijk(scenario, vehicle_types, i, j, k,)
                M_ijk = b_i + s_i + t_ijk
                model += (T_var[j, k] >= T_var[i, k] + s_i + t_ijk - M_ijk * (1 - x[i, j, k]), f"time_continuity_{i}_{j}_{k}")

                if get_a_i(scenario, i) + s_i + t_ijk > get_b_i(scenario, j):
                    model += x[i, j, k] == 0, f"time_incompatible_{i}_{j}_{k}"

    # depot departure and arrival time constraints for vehicle k
    # (purpose to include P,R)
    for k in K:
        depot = get_h(k)
        vehicle_type = get_tau(k)
        latest_return = get_B_t_return(vehicle_types, vehicle_type)
        for i in C:
            t_i_depot = get_t_ijk(scenario, vehicle_types, i, depot, k,)
            t_depot_i = get_t_ijk(scenario, vehicle_types, depot, i, k,)
            s_i = get_s_i(scenario, i)
            M_i_depot = get_b_i(scenario, i) + s_i + t_i_depot
            M_depot_i = latest_return + t_depot_i
            model += (T_var[i, k] >= P[k] + t_depot_i - M_depot_i * (1 - x[depot, i, k]), f"time_from_depot_{i}_{k}")
            model += (R[k] >= T_var[i, k] + s_i + t_i_depot - M_i_depot * (1 - x[i, depot, k]), f"time_to_depot_{i}_{k}")

    # vehicle availability constraints Pk,Rk
    for k in K:
        vehicle_type = get_tau(k)
        A_t_start = get_A_t_start(vehicle_types, vehicle_type)
        B_t_return = get_B_t_return(vehicle_types, vehicle_type)
        H_t = get_H_t(vehicle_types, vehicle_type) # max route duration

        model += P[k] >= A_t_start * z[k], f"start_after_availability_{k}"
        model += P[k] <= B_t_return * z[k], f"start_deadline_{k}"
        model += R[k] >= A_t_start * z[k], f"return_after_availability_{k}"
        model += R[k] <= B_t_return * z[k], f"return_deadline_{k}"
        model += R[k] - P[k] <= H_t * z[k], f"max_duration_{k}"

def build_model(scenario, vehicle_types):
    model = pulp.LpProblem("VRP", pulp.LpMinimize)

    # build sets
    # sort for reproducibility of results
    C = sorted(get_C(scenario))
    D = sorted(get_D(scenario))
    T = sorted(get_T(scenario))
    K = sorted(get_K(scenario))


    # decision variables
    x_ijk = []

    for k in K:
        for i, j in get_A_k(scenario, k):
            x_ijk.append((i, j, k))
    x = pulp.LpVariable.dicts("x", x_ijk, lowBound=0, cat="Binary")

    y_ik = []

    for k in K: 
        for i in C:
            y_ik.append((i,k))
    y = pulp.LpVariable.dicts("y", y_ik, lowBound=0, cat="Binary")

    z_k = []
    for k in K: 
        z_k.append(k)
    z = pulp.LpVariable.dicts("z", z_k, lowBound=0, cat="Binary")

    u_ik = []
    for i in C:
        for k in K: 
            u_ik.append((i,k))
    u = pulp.LpVariable.dicts("u", u_ik, lowBound=0, cat="Continuous")

    T_ik = []
    for i in C:
        for k in K: 
            T_ik.append((i,k))
    T_var = pulp.LpVariable.dicts("T_start", T_ik, lowBound=0, cat="Continuous")

    P_k = []
    for k in K: 
        P_k.append(k)
    P = pulp.LpVariable.dicts("P", P_k, lowBound=0, cat="Continuous")

    R_k = []
    for k in K: 
        R_k.append(k)
    R = pulp.LpVariable.dicts("R", R_k, lowBound=0, cat="Continuous")

    # objective
    add_objective(model, scenario, vehicle_types, x, z)

    # constraints
    add_core_constraints(model, scenario, x, y, z)
    add_capacity_constraints(model, scenario, vehicle_types, y, z)
    if scenario.get("use_time_constraints", True):
        # Positive travel/service times already eliminate customer subtours.
        add_time_window_constraints(model, scenario, vehicle_types, x, y, z, T_var, P, R)
    else:
        add_subtour_constraints(model, scenario, vehicle_types, x, y, u)

    variables = {
        "x": x,
        "y": y,
        "z": z,
        "u": u,
        "T": T_var,
        "P": P,
        "R": R
    }

    return model, variables
