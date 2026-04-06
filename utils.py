# utils.py

import matplotlib.pyplot as plt
from data import points

def plot_route(route, title):
    if route is None:
        print(f"No route for {title}")
        return

    x = [points[p][0] for p in route]
    y = [points[p][1] for p in route]

    plt.figure()
    plt.plot(x, y, marker='o')

    for name, (px, py) in points.items():
        plt.text(px, py, name, fontsize=8)

    plt.title(title)
    plt.grid()
    plt.show()