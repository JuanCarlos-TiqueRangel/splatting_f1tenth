#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

# Put your 4 workspace corners here.
# They can be in any order.
# Format: [x, y]

    # [-2.5, -1.5],
    # [2.5, -1.5],
    # [2.5, 1.5],
    # [-2.5, 1.5]

workspace_corners = np.array([
    [-1.8, -0.9],
    [1.8, -0.9],
    [1.8, 0.9],
    [-1.8, 0.9]
], dtype=float)

# Distance from the wall to the generated trajectory centerline
safety_margin = 0.10   # meters

# Sampling distance between consecutive trajectory points
ds = 0.05              # meters

# These are the columns in your example CSV
w_tr_right_m = 0.70
w_tr_left_m  = 0.70

# Output CSV name
output_csv = "generated_square_trajectory_small.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def polygon_signed_area(points):
    """Positive if points are in CCW order."""
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))


def order_points_clockwise_or_ccw(points):
    """
    Order 4 points around their centroid.
    Returns them in CCW order.
    """
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]

    # Make sure it is CCW
    if polygon_signed_area(ordered) < 0:
        ordered = ordered[::-1]

    return ordered


def line_intersection(p1, p2, p3, p4):
    """
    Intersection of two infinite lines:
    line 1 through p1-p2
    line 2 through p3-p4
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)

    if abs(denom) < 1e-12:
        raise ValueError("Two offset edges are parallel or nearly parallel.")

    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom

    return np.array([px, py])


def offset_polygon_inward(points, offset):
    """
    Offset a convex polygon inward by 'offset'.
    Assumes points are in CCW order.
    """
    n = len(points)
    new_points = []

    offset_lines = []

    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]

        edge = p2 - p1
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-12:
            raise ValueError("Two workspace corners are identical or too close.")

        # For CCW polygon, inward normal is to the left of the edge
        inward_normal = np.array([-edge[1], edge[0]]) / edge_len

        q1 = p1 + offset * inward_normal
        q2 = p2 + offset * inward_normal
        offset_lines.append((q1, q2))

    for i in range(n):
        q1, q2 = offset_lines[i]
        q3, q4 = offset_lines[(i + 1) % n]
        inter = line_intersection(q1, q2, q3, q4)
        new_points.append(inter)

    return np.array(new_points)


def sample_segment(p1, p2, ds):
    """Sample points from p1 to p2, excluding p2."""
    vec = p2 - p1
    length = np.linalg.norm(vec)

    if length < 1e-12:
        return np.empty((0, 2))

    n_steps = max(1, int(np.floor(length / ds)))
    t_values = np.linspace(0.0, 1.0, n_steps, endpoint=False)
    pts = p1[None, :] + t_values[:, None] * vec[None, :]
    return pts


def sample_closed_polygon(points, ds):
    """Sample the full closed polygon without duplicating the last point."""
    sampled = []
    n = len(points)

    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        seg_pts = sample_segment(p1, p2, ds)
        if len(seg_pts) > 0:
            sampled.append(seg_pts)

    if len(sampled) == 0:
        return np.empty((0, 2))

    return np.vstack(sampled)


def validate_workspace(points):
    if points.shape != (4, 2):
        raise ValueError("You must provide exactly 4 corners with shape (4, 2).")

    # Check that the inward offset is feasible later
    ordered = order_points_clockwise_or_ccw(points)

    # Side lengths
    lengths = []
    for i in range(4):
        lengths.append(np.linalg.norm(ordered[(i + 1) % 4] - ordered[i]))
    lengths = np.array(lengths)

    if np.min(lengths) <= 2.0 * safety_margin:
        raise ValueError(
            "Safety margin is too large for this workspace. "
            "Decrease safety_margin."
        )

    return ordered


# ============================================================
# MAIN
# ============================================================

def main():
    outer = validate_workspace(workspace_corners)

    # Create inner square/rectangle trajectory
    inner = offset_polygon_inward(outer, safety_margin)

    # Sample points along the inner loop
    traj_xy = sample_closed_polygon(inner, ds)

    # Build DataFrame in the same style as your trajectory file
    df = pd.DataFrame({
        "# x_m": traj_xy[:, 0],
        "y_m": traj_xy[:, 1],
        "w_tr_right_m": np.full(len(traj_xy), w_tr_right_m),
        "w_tr_left_m": np.full(len(traj_xy), w_tr_left_m),
    })

    # Save CSV
    df.to_csv(output_csv, index=False)
    print(f"Trajectory saved to: {output_csv}")
    print(f"Number of trajectory points: {len(df)}")

    # Plot
    plt.figure(figsize=(8, 8))

    # Outer workspace
    outer_closed = np.vstack([outer, outer[0]])
    plt.plot(outer_closed[:, 0], outer_closed[:, 1], "k--", label="Workspace boundary")

    # Inner generated trajectory
    inner_closed = np.vstack([inner, inner[0]])
    plt.plot(inner_closed[:, 0], inner_closed[:, 1], "b-", linewidth=2, label="Generated trajectory")
    plt.scatter(traj_xy[:, 0], traj_xy[:, 1], s=8, label="Sampled points")

    # Draw original corners
    plt.scatter(outer[:, 0], outer[:, 1], s=60, marker="o", label="Input corners")

    for i, p in enumerate(outer):
        plt.text(p[0], p[1], f"  P{i+1}", fontsize=10)

    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Generated Square Trajectory from Workspace Boundaries")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()