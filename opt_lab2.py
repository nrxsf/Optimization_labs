import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from math import ceil, sqrt
from opt_lab1 import golden_search


def G(x1, x2):
    return 0.5 * ((x1 ** 4 - 16 * x1 ** 2 + 5 * x1) + (x2 ** 4 - 16 * x2 ** 2 + 5 * x2))


def numerical_gradient(f, x1, x2, h=0.0001):
    df_dx1 = (f(x1 + h, x2) - f(x1 - h, x2)) / (2 * h)
    df_dx2 = (f(x1, x2 + h) - f(x1, x2 - h)) / (2 * h)
    return [df_dx1, df_dx2]


def classic_gradient_descent(x0, f, eps=1e-6, max_iter=500):
    start_time = time.perf_counter()
    alpha = 0.05
    path = [np.array(x0)]
    iters = 0
    func_evals = 0

    for i in range(max_iter):
        iters += 1
        x_prev = path[-1]
        grad = np.array(numerical_gradient(f, *x_prev))
        func_evals += 4

        x_next = x_prev - alpha * grad
        path.append(x_next)

        if np.linalg.norm(x_next - x_prev) < eps:
            break

    time_taken = time.perf_counter() - start_time
    return np.array(path), iters, time_taken, func_evals


def cauchy_gradient_descent(x0, f, eps=1e-6, max_iter=500):
    start_time = time.perf_counter()
    path = [np.array(x0)]
    iters = 0
    func_evals = 0

    for i in range(max_iter):
        iters += 1
        x_prev = path[-1]
        grad = np.array(numerical_gradient(f, *x_prev))

        func = lambda a: f(x_prev[0] - grad[0] * a, x_prev[1] - grad[1] * a)
        alpha, gs_evals = golden_search(func, 0, 0.5, epsilon=1e-6)

        func_evals += gs_evals
        x_next = x_prev - alpha * grad
        path.append(x_next)

        if np.linalg.norm(x_next - x_prev) < eps:
            break

    time_taken = time.perf_counter() - start_time
    return np.array(path), iters, time_taken, func_evals


def hooke_jeeves(x0, f, h=0.5, eps=1e-6, alpha=2.0, max_iter=500):
    start_time = time.perf_counter()
    x_curr = np.array(x0, dtype=float)
    f_curr = f(*x_curr)
    path = [x_curr.copy()]
    func_evals = 1
    iters = 0

    while iters < max_iter:
        iters += 1
        x_before_expl = x_curr.copy()

        for i in range(len(x_curr)):
            old_val = x_curr[i]
            x_curr[i] = old_val + h
            f_next = f(*x_curr)
            func_evals += 1

            if f_next < f_curr:
                f_curr = f_next
                path.append(x_curr.copy())
            else:
                x_curr[i] = old_val - h
                f_next = f(*x_curr)
                func_evals += 1
                if f_next < f_curr:
                    f_curr = f_next
                    path.append(x_curr.copy())
                else:
                    x_curr[i] = old_val

        if not np.array_equal(x_curr, x_before_expl):
            while True:
                x_3 = x_curr.copy()
                direction = x_curr - x_before_expl
                for i in range(len(x_3)):
                    if direction[i] != 0:
                        x_3[i] += direction[i]
                        path.append(x_3.copy())

                f_3 = f(*x_3)
                func_evals += 1

                if f_3 < f_curr:
                    x_before_expl = x_curr.copy()
                    x_curr = x_3.copy()
                    f_curr = f_3
                else:
                    break
        else:
            if h < eps:
                break
            h /= alpha

    time_taken = time.perf_counter() - start_time
    return np.array(path), iters, time_taken, func_evals


def build_simplex(x0, size):
    dots = []
    n = len(x0)
    delta1 = (sqrt(n + 1) + n - 1) / (n * sqrt(2)) * size
    delta2 = (sqrt(n + 1) - 1) / (n * sqrt(2)) * size

    for i in range(n):
        new_dot = x0.copy()
        for j in range(n):
            new_dot[j] += delta2 if i == j else delta1
        dots.append(new_dot)
    return dots


def simplex_method(x0, f, alpha=0.5, reduction=0.5, repeat=2, stretch=3, squeeze=1.5, eps=1e-6, max_iter=500):
    start_time = time.perf_counter()
    iters, func_eval, n = 0, 0, len(x0)
    m = ceil(1.65 * n + 0.05 * n ** 2)

    simplex_base = np.array(x0, dtype=float)
    f_base = f(*simplex_base)
    func_eval += 1

    dots = [[simplex_base, f_base, 0]]
    initial_coords = build_simplex(simplex_base, alpha)

    for coords in initial_coords:
        dots.append([coords, f(*coords), 0])
        func_eval += 1

    path = []

    while True:
        iters += 1
        dots.sort(key=lambda x: x[1])
        best_dot = dots[0]
        worst_dot = dots[-1]

        current_coords = [d[0].copy() for d in dots]
        current_coords.append(dots[0][0].copy())
        path.append(current_coords)

        max_life = 0
        for d in dots:
            d[2] += 1
            max_life = max(max_life, d[2])

        if max_life > m + 1:
            alpha *= reduction
            new_base = best_dot[0]
            dots = [[new_base, best_dot[1], 0]]
            for coords in build_simplex(new_base, alpha):
                dots.append([coords, f(*coords), 0])
                func_eval += 1
            continue

        all_coords = np.array([d[0] for d in dots])
        x_centre = (np.sum(all_coords, axis=0) - worst_dot[0]) / n

        x_normal = worst_dot[0] + repeat * (x_centre - worst_dot[0])
        x_stretched = worst_dot[0] + stretch * (x_centre - worst_dot[0])
        x_squeezed = worst_dot[0] + squeeze * (x_centre - worst_dot[0])

        f_n, f_st, f_sq = f(*x_normal), f(*x_stretched), f(*x_squeezed)
        func_eval += 3

        candidates = [(x_normal, f_n), (x_stretched, f_st), (x_squeezed, f_sq)]
        best_cand_coords, best_cand_f = min(candidates, key=lambda x: x[1])

        dots[-1] = [best_cand_coords, best_cand_f, 0]

        if iters > max_iter or max(np.linalg.norm(d[0] - best_dot[0]) for d in dots) < eps:
            break

    time_taken = time.perf_counter() - start_time
    return np.array(path), iters, time_taken, func_eval


if __name__ == '__main__':
    n = 4
    x_val = np.linspace(-n, n, 50)
    y_val = np.linspace(-n, n, 50)
    X, Y = np.meshgrid(x_val, y_val)
    Z = G(X, Y)

    start_p = [np.random.uniform(-n, n), np.random.uniform(-4, 4)]

    path_classic, iters_classic, time_classic, evals_classic = classic_gradient_descent(start_p, G)
    path_cauchy, iters_cauchy, time_cauchy, evals_cauchy = cauchy_gradient_descent(start_p, G)
    path_hj, iters_hj, time_hj, evals_hj = hooke_jeeves(start_p, G, h=0.4)
    path_simplex, iters_simplex, time_simplex, evals_simplex = simplex_method(start_p, G)

    fig = plt.figure(figsize=(18, 6))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', zorder=1)

    ax1.plot(path_classic[:, 0], path_classic[:, 1], G(path_classic[:, 0], path_classic[:, 1]),
             'r.-', label='Classic GD', zorder=10)
    ax1.plot(path_cauchy[:, 0], path_cauchy[:, 1], G(path_cauchy[:, 0], path_cauchy[:, 1]),
             'b.-', label='Cauchy GD', zorder=11)

    ax1.set_title("Gradient-Based Methods")
    ax1.legend()

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, Z, cmap='viridis', zorder=1)

    ax2.plot(path_hj[:, 0], path_hj[:, 1], G(path_hj[:, 0], path_hj[:, 1]),
             'g.-', label='Hooke-Jeeves', zorder=12)

    for i, simplex in enumerate(path_simplex):
        label = 'Simplex Method' if i == 0 else ""
        sx, sy = [p[0] for p in simplex], [p[1] for p in simplex]
        ax2.plot(sx, sy, G(np.array(sx), np.array(sy)),
                 color='magenta', linestyle='-', marker='.', label=label, zorder=13)

    ax2.set_title("Direct Search Methods")
    ax2.legend()

    ax3 = fig.add_subplot(133)
    contour = ax3.contourf(X, Y, Z, cmap='viridis', levels=40)
    plt.colorbar(contour, ax=ax3)

    ax3.plot(path_classic[:, 0], path_classic[:, 1], 'r-', label='Classic')
    ax3.plot(path_cauchy[:, 0], path_cauchy[:, 1], 'b-', label='Cauchy')
    ax3.plot(path_hj[:, 0], path_hj[:, 1], 'g-', label='H-J')

    for i, simplex in enumerate(path_simplex):
        sx, sy = [p[0] for p in simplex], [p[1] for p in simplex]
        ax3.plot(sx, sy, color='magenta', linestyle='-', alpha=0.5)

    ax3.scatter(*start_p, color='yellow', edgecolors='black', s=70, label='Start', zorder=15)
    ax3.scatter(path_classic[-1, 0], path_classic[-1, 1], color='red', marker='X', s=100, label='End Classic',
                zorder=16)
    ax3.scatter(path_cauchy[-1, 0], path_cauchy[-1, 1], color='blue', marker='X', s=100, label='End Cauchy', zorder=16)

    ax3.set_title("2D Path Comparison")
    ax3.legend()

    summary_data = {
        "Method": ["Classic Gradient Descent", "Steepest Descent (Cauchy)", "Hooke-Jeeves", "Simplex Method"],
        "Execution Time (s)": [time_classic, time_cauchy, time_hj, time_simplex],
        "Iterations": [iters_classic, iters_cauchy, iters_hj, iters_simplex],
        "Func Evaluations": [evals_classic, evals_cauchy, evals_hj, evals_simplex]
    }

    df_summary = pd.DataFrame(summary_data)
    print("\n" + "=" * 80 + "\nPerformance Summary Table:\n" + "=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80 + "\n")

    plt.show()
