import matplotlib.pyplot as plt

file = open("data", "r")

points = [ point.strip().split(' ') for point in file.read().strip().split('\n') ]
points = [ tuple([ float(coord) for coord in point ]) for point in points ]

num_ys = len(points[0]) - 1


plots = []

def plot_all():
    fig, axs = plt.subplots(len(plots))
    if num_ys == 1:
        axs = [axs]

    for (i, (xs, ys, title)) in enumerate(plots):
        axs[i].plot(xs, ys)
        axs[i].title.set_text(title)

    plt.show()
    plots.clear() # clear plots


xs = [ point[0] for point in points ]
for i in range(num_ys):
    ys = [ point[i + 1] for point in points ]
    plots.append((xs, ys, f"$y_{i}(t)$"))

plot_all()

for i in range(num_ys):
    xs = [ point[i + 1] for point in points ]
    for j in range(i + 1, num_ys):
        ys = [ point[j + 1] for point in points ]
        plots.append((xs, ys, f"$y_{j}(y_{i})$"))

plot_all()
