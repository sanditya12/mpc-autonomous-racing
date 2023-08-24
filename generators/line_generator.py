import numpy as np


line_segments = [
    [[1.0999999999999996, 12.45], [1.0999999999999996, 4.45]]
    # [
    #     [150, 250],
    #     [400, 100],
    # ],
    # [[400, 100], [600, 100]],
    # [[600, 100], [850, 250]],
    # [[850, 250], [850, 350]],
    # [[850, 350], [600, 500]],
    # [[600, 500], [400, 500]],
    # [[400, 500], [150, 350]],
    # [[150, 350], [150, 250]],
]

sample_points = []
for segment in line_segments:
    (x1, y1), (x2, y2) = segment
    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    theta = np.arctan2((y2 - y1), (x2 - x1))

    for dist in np.arange(0, length, 0.15):
        x_new = x1 + dist * np.cos(theta)
        y_new = y1 + dist * np.sin(theta)
        sample_points.append((x_new, y_new))

for p in sample_points:
    # print('{"xc": ', p[0], ', "yc": ', p[1], "},")
    print("[ ", p[0], " , ", p[1], " ],")
