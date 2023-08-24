import numpy as np


def quarter_circle_points(center, radius, spacing, quarter="RT"):
    """
    Computes the evenly spaced points for a quarter circle.

    Parameters:
    - center: tuple (x, y) denoting the center of the circle.
    - radius: the radius of the circle.
    - spacing: the desired spacing distance between points.
    - quarter: string indicating which quarter of the circle. Options:
        'RT' (right-top), 'LT' (left-top), 'LB' (left-bottom), 'RB' (right-bottom).

    Returns:
    - List of (x, y) points for the desired quarter circle segment.
    """

    # Determine the starting and ending angles based on the chosen quarter
    angles = {
        "RT": (0, np.pi / 2),
        "LT": (np.pi / 2, np.pi),
        "LB": (np.pi, 3 * np.pi / 2),
        "RB": (3 * np.pi / 2, 2 * np.pi),
    }

    start_angle, end_angle = angles[quarter]

    # Number of points can be estimated by arc length (which is a quarter of the circumference)
    # divided by the spacing. This gives a rough estimate which can be adjusted.
    num_points = int(radius * np.pi / 2 / spacing) + 1

    # linspace generates evenly spaced numbers over a specified range.
    theta = np.linspace(start_angle, end_angle, num_points)

    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)

    return list(zip(x, y))


arc_points = quarter_circle_points([4.1, 4.5], 1.5, 0.15, "LB")

for p in arc_points:
    print("[ ", p[0], " , ", p[1], " ],")
