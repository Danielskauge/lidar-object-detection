def calculate_actual_power_usage(power_usage_percent, max_power_watts):
    """Calculate actual power usage in watts from power usage percentage."""
    return (power_usage_percent / 100) * max_power_watts

def calculate_energy(power_watts, duration_hours):
    """Calculate total energy consumption in kWh."""
    total_energy_wh = power_watts * duration_hours * 60  # Convert hours to minutes
    total_energy_kwh = total_energy_wh / 1000.0
    return total_energy_kwh

def calculate_driving_distance(total_energy_kwh, efficiency_wh_per_km=150):
    """Calculate how far a Tesla Model 3 could drive using the given energy."""
    distance_km = total_energy_kwh * 1000 / efficiency_wh_per_km
    return distance_km

# GPU power specifications for GeForce RTX 4090
max_power_watts = 450  # Max power in watts for GeForce RTX 4090

# Training specifics
power_usage_percent = 30  # 30% GPU power utilization
training_duration_minutes = 35  # 35 minutes of training

# Calculate actual power usage
actual_power_usage_watts = calculate_actual_power_usage(power_usage_percent, max_power_watts)

# Calculate total energy consumption
total_energy_kwh = calculate_energy(actual_power_usage_watts, training_duration_minutes / 60)

# Calculate driving distance for a Tesla Model 3
driving_distance_km = calculate_driving_distance(total_energy_kwh)

print(f"Driving distance using the energy consumed during training: {driving_distance_km:.2f} km")