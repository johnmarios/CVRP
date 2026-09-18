import math


def capacity_precheck(demands, number_of_vehicles, capacity):
    """Check simple necessary capacity conditions before calling the solver."""
    total_demand = sum(demands)
    aggregate_lower_bound = math.ceil(total_demand / capacity)
    large_customer_lower_bound = sum(demand > capacity / 2 for demand in demands)
    capacity_lower_bound = max(aggregate_lower_bound, large_customer_lower_bound)

    reason = None

    if max(demands) > capacity:
        reason = (
            f"Maximum customer demand {max(demands)} exceeds "
            f"vehicle capacity {capacity}."
        )
    elif aggregate_lower_bound > number_of_vehicles:
        reason = (
            f"Total demand {total_demand} exceeds fleet capacity "
            f"{number_of_vehicles} x {capacity} = {number_of_vehicles * capacity}."
        )
    elif large_customer_lower_bound > number_of_vehicles:
        reason = (
            f"{large_customer_lower_bound} customers have demand greater "
            f"than Q/2={capacity / 2:g}; they cannot share a vehicle, "
            f"but only {number_of_vehicles} vehicles are available."
        )

    return {
        "infeasible": reason is not None,
        "aggregate_lower_bound": aggregate_lower_bound,
        "large_customer_lower_bound": large_customer_lower_bound,
        "capacity_lower_bound": capacity_lower_bound,
        "reason": reason or "The necessary capacity checks pass.",
    }
