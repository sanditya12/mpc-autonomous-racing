from track_data import center_points


def find_boundaries(center_points, width):
    left_boundary = []
    right_boundary = []
    n = len(center_points)

    # For each center point
    for i in range(n):
        if (
            i == 0
        ):  # First point: use the difference between the first and second point
            direction = [
                center_points[i + 1][0] - center_points[i][0],
                center_points[i + 1][1] - center_points[i][1],
            ]
        elif (
            i == n - 1
        ):  # Last point: use the difference between the last and second to last point
            direction = [
                center_points[i][0] - center_points[i - 1][0],
                center_points[i][1] - center_points[i - 1][1],
            ]
        else:  # Middle points: average the difference between next and previous points
            direction = [
                (center_points[i + 1][0] - center_points[i - 1][0]) / 2,
                (center_points[i + 1][1] - center_points[i - 1][1]) / 2,
            ]

        # Normalize the direction vector
        magnitude = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
        direction = [direction[0] / magnitude, direction[1] / magnitude]

        # Compute the normal vector (rotate the direction vector by 90 degrees)
        normal = [-direction[1], direction[0]]

        # Calculate left and right boundary points
        left_boundary.append(
            [
                center_points[i][0] + normal[0] * width / 2,
                center_points[i][1] + normal[1] * width / 2,
            ]
        )
        right_boundary.append(
            [
                center_points[i][0] - normal[0] * width / 2,
                center_points[i][1] - normal[1] * width / 2,
            ]
        )

    return left_boundary, right_boundary


left, right = find_boundaries(center_points, 3)

print("Left Boundary:")
print(left)

print("\nRight Boundary:")
print(right)
