import math
import time

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pydobot import Dobot
from serial.tools import list_ports

# Accurate dimensions from the specifications
base_height = 138  # Height of the base
rear_arm_length = 135  # Length of the rear arm
forearm_length = 147  # Length of the forearm
max_reach = 320  # Maximum reach as specified
MIN_REACH = 135

# Calculate the center of the workspace
center_x = max_reach / 2
center_y = 0
center_z = base_height + (max_reach / 2)

center = (center_x, center_y, center_z)
radius = max_reach / 2

print(f"Estimated center of workspace: {center}")


def generate_sphere_points(center, radius, num_points=20000):
    phi = np.random.uniform(0, np.pi, num_points)
    theta = np.random.uniform(0, 2 * np.pi, num_points)

    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)

    return x, y, z


def apply_constraints(x, y, z, base_height, rear_arm_length):
    # Adjust coordinates to robot's base coordinate system
    x_adj = x - center[0]
    y_adj = y - center[1]
    z_adj = z - base_height

    # Apply constraints
    distance_from_base = np.sqrt(x_adj**2 + y_adj**2 + z_adj**2)
    mask = (z >= base_height) & (distance_from_base <= max_reach) & (distance_from_base >= rear_arm_length)

    # Apply motion range constraints
    base_angle = np.arctan2(y_adj, x_adj) * 180 / np.pi
    mask &= (base_angle >= -90) & (base_angle <= 90)

    return x[mask], y[mask], z[mask]


def backup():
    # Generate points
    x, y, z = generate_sphere_points(center, radius)

    # Apply constraints
    x, y, z = apply_constraints(x, y, z, base_height, rear_arm_length)

    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot points
    scatter = ax.scatter(x, y, z, c=z, cmap="viridis", marker=".", s=1)

    # Plot the base and arm
    ax.plot([0, 0], [0, 0], [0, base_height], color="black", linewidth=2)
    ax.plot([0, rear_arm_length], [0, 0], [base_height, base_height], color="blue", linewidth=2)
    ax.plot([rear_arm_length, max_reach], [0, 0], [base_height, base_height], color="green", linewidth=2)

    # Set labels and title
    ax.set_xlabel("X axis (mm)")
    ax.set_ylabel("Y axis (mm)")
    ax.set_zlabel("Z axis (mm)")
    ax.set_title("Workspace of Dobot Magician")

    # Set equal aspect ratio
    ax.set_box_aspect((1, 1, 1))

    # Add a color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Height (mm)")

    # Add text annotations
    ax.text2D(0.05, 0.95, f"Max reach: {max_reach}mm", transform=ax.transAxes)
    ax.text2D(0.05, 0.90, f"Base height: {base_height}mm", transform=ax.transAxes)
    ax.text2D(0.05, 0.85, f"Rear arm: {rear_arm_length}mm", transform=ax.transAxes)
    ax.text2D(0.05, 0.80, f"Forearm: {forearm_length}mm", transform=ax.transAxes)

    plt.show()


def connect_to_dobot():
    port = list_ports.comports()[0].device
    print(f"Connecting to Dobot on port {port}...")
    device = Dobot(port=port, verbose=True)
    time.sleep(2)  # Wait for the connection to stabilize
    return device


def move_to_point(device, x, y, z):
    print(f"Moving to point: x={x:.2f}, y={y:.2f}, z={z:.2f}")
    device.move_to(x, y, z, 0, wait=True)
    time.sleep(0.5)  # Short pause at each point


def trace_workspace_border(device):
    print("Tracing workspace border...")
    # Trace the outer circle at various heights
    heights = [base_height, base_height + 50, base_height + 100, base_height + 150]
    for height in heights:
        print(f"Tracing circle at height {height}mm")
        for angle in range(0, 181, 10):  # -90 to 90 degrees
            rad = math.radians(angle - 90)
            x = max_reach * math.cos(rad)
            y = max_reach * math.sin(rad)
            move_to_point(device, x, y, height)

    # Trace vertical lines at various angles
    angles = [-90, -45, 0, 45, 90]
    for angle in angles:
        print(f"Tracing vertical line at angle {angle} degrees")
        rad = math.radians(angle)
        for height in range(base_height, base_height + max_reach, 20):
            x = max_reach * math.cos(rad)
            y = max_reach * math.sin(rad)
            move_to_point(device, x, y, height)

    # Trace the inner circle (minimum reach) at base height
    print(f"Tracing inner circle at height {base_height}mm")
    for angle in range(0, 361, 10):
        rad = math.radians(angle - 90)
        x = MIN_REACH * math.cos(rad)
        y = MIN_REACH * math.sin(rad)
        move_to_point(device, x, y, base_height)


def run_workspace_trace():
    device = None

    try:
        # Connect to the Dobot
        device = connect_to_dobot()

        # Get the initial position
        x, y, z, r, j1, j2, j3, j4 = device.pose()
        print(f"Initial position: x={x:.2f}, y={y:.2f}, z={z:.2f}")

        # Trace the workspace border
        trace_workspace_border(device)

        # Return to the initial position
        print("Returning to initial position...")
        device.move_to(x, y, z, r, wait=True)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Disconnect from the Dobot
        print("Disconnecting from Dobot...")
        if device:
            device.close()


if __name__ == "__main__":
    print("Starting Dobot Magician workspace tracing program...")
    run_workspace_trace()
    print("Program completed.")
