import numpy as np
from track_data import center_sample_points_v2

center_points = center_sample_points_v2.center_points

import math


def get_line_equation(p1, p2):
    """
    Return the slope m and y-intercept c of the line passing through points p1 and p2.
    If the line is vertical, return None for the slope.
    """
    if p1[0] == p2[0]:  # vertical line
        return None, p1[0]
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    c = p1[1] - m * p1[0]
    return m, c


def get_arc_center_and_radius(p1, p2, p3):
    """
    Return the center and the radius of the arc passing through points p1, p2, and p3.
    """
    # Midpoints
    D = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    E = ((p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2)

    # Slopes for the perpendicular bisectors
    if p2[0] - p1[0] != 0:
        m1 = -(p2[1] - p1[1]) / (p2[0] - p1[0])
    else:
        m1 = None

    if p3[0] - p2[0] != 0:
        m2 = -(p3[1] - p2[1]) / (p3[0] - p2[0])
    else:
        m2 = None

    # Equations of the perpendicular bisectors
    if m1 is not None:
        c1 = D[1] - m1 * D[0]
    if m2 is not None:
        c2 = E[1] - m2 * E[0]

    # Check if the lines are parallel
    if m1 == m2:
        raise ValueError(
            f"The points {p1}, {p2}, and {p3} are collinear or lead to parallel bisectors."
        )

    # Intersection of the bisectors gives the center
    if m1 is None:
        Ox = D[0]
        Oy = m2 * Ox + c2
    elif m2 is None:
        Ox = E[0]
        Oy = m1 * Ox + c1
    else:
        Ox = (c2 - c1) / (m1 - m2)
        Oy = m1 * Ox + c1

    # Radius is the distance from the center to one of the points
    r = math.sqrt((Ox - p1[0]) ** 2 + (Oy - p1[1]) ** 2)

    return (Ox, Oy), r


def preprocess_center_points(center_points):
    segments = []
    i = 0
    while i < len(center_points) - 1:
        # Check if this is a line segment (2 consecutive points)
        if (
            i == len(center_points) - 2
            or math.isclose(center_points[i][0], center_points[i + 2][0])
            or math.isclose(center_points[i][1], center_points[i + 2][1])
        ):
            m, c = get_line_equation(center_points[i], center_points[i + 1])
            segments.append(("line", (m, c)))
            i += 1
        # Otherwise, it's an arc segment (3 consecutive points)
        else:
            center, radius = get_arc_center_and_radius(
                center_points[i], center_points[i + 1], center_points[i + 2]
            )
            segments.append(("arc", (center, radius)))
            i += 2
    return segments


segments = preprocess_center_points(center_points)
for s in segments:
    print(s)
    print(" ")
