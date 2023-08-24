from track_data import center_sample_points

center_points = center_sample_points.center_points


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def find_projection_on_segment(pos, start, end):
    # distance between start of line segment and end of segment
    vec1 = [end[0] - start[0], end[1] - start[1]]

    # distance between start of line segment and positon
    vec2 = [pos[0] - start[0], pos[1] - start[1]]
    # print(vec1)
    vec1_squared = dot(vec1, vec1)
    vec1_vec2 = dot(vec1, vec2)

    # t = 0 -> projection = start
    t = min(1, max(0, vec1_vec2 / vec1_squared))
    # The minmax is used to not let the projection that goes beyond point end or point start

    return [start[0] + vec1[0] * t, start[1] + vec1[1] * t]


def find_projection_to_center(position, center_points):
    min_distance = float("inf")
    closest_point = None

    for i in range(len(center_points) - 1):
        a = center_points[i]
        b = center_points[i + 1]

        projected_point = find_projection_on_segment(position, a, b)

        distance = (
            (position[0] - projected_point[0]) ** 2
            + (position[1] - projected_point[1]) ** 2
        ) ** 0.5

        if distance < min_distance:
            min_distance = distance
            closest_point = projected_point

    return closest_point


# robot_position = [1.5, 2.0]
# projected_point = find_projection_to_center(robot_position, center_points)

# print("Robot's position:", robot_position)
# print("Projection onto center line:", projected_point)
