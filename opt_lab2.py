import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from math import ceil, sqrt
from opt_lab1 import golden_search


def G(x1, x2):
    return 0.5 * ((x1 ** 4 - 16 * x1 ** 2 + 5 * x1) + (x2 ** 4 - 16 * x2 ** 2 + 5 * x2))


def G_x1(x1, x2):
    return 0.5 * (4 * x1 ** 3 - 32 * x1 + 5)


def G_x2(x1, x2):
    return 0.5 * (4 * x2 ** 3 - 32 * x2 + 5)


def analytical_gradient(f_x, f_y, x1, x2):
    return [f_x(x1, x2), f_y(x1, x2)]


def numerical_gradient(f, x1, x2, h=0.0001):
    return [((f(x1 + h, x2) - f(x1 - h, x2)) / (2 * h)), ((f(x1, x2 + h) - f(x1, x2 - h)) / (2 * h))]


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


def koshi_gradient_descent(x0, f, f_x, f_y, eps=1e-6, max_iter=500):
    start_time = time.perf_counter()
    path = [np.array(x0)]

    iters = 0
    func_evals = 0

    for i in range(max_iter):
        iters += 1
        x_prev = path[-1]
        grad = np.array(analytical_gradient(f_x, f_y, *x_prev))

        x1 = lambda a: x_prev[0] - grad[0] * a
        x2 = lambda a: x_prev[1] - grad[1] * a

        func = lambda a: f(x1(a), x2(a))

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
            if i == j:
                new_dot[j] += delta2
            else:
                new_dot[j] += delta1
        dots.append(new_dot)
    return dots


def simplex_method(x0, f, alpha=0.5, reduction=0.5, repeat=2, stretch=3, squeeze=1.5, eps=1e-6, max_iter=500):
    start_time = time.perf_counter()
    iters = 0
    func_eval = 0
    n = len(x0)
    m = ceil(1.65 * n + 0.05 * n ** 2)
    simplex_base = np.array(x0.copy(), dtype=float)
    path = []

    dots = [simplex_base]
    dots_life = [0] * (n + 1)
    dots.extend(build_simplex(simplex_base, alpha))

    best_x = simplex_base

    while True:
        iters += 1

        if max(dots_life) > m + 1:
            alpha *= reduction
            dots.clear()

            dots.append(best_x)
            dots.extend(build_simplex(best_x, alpha))

            dots_life = [0] * (n + 1)

        for i in range(n + 1):
            dots_life[i] += 1

        dots_copy = [d.copy() for d in dots]
        dots_copy.append(dots[0].copy())
        path.append(dots_copy)

        worse_index = 0
        worse_x = dots[0]
        best_x = dots[0]

        worse_f = f(*dots[0])
        best_f = worse_f
        func_eval += 1

        for i in range(1, n + 1):
            f_curr = f(*dots[i])
            func_eval += 1
            if f_curr > worse_f:
                worse_x = dots[i]
                worse_f = f_curr
                worse_index = i
            if f_curr < best_f:
                best_x = dots[i]
                best_f = f_curr

        dots_life[worse_index] = 0
        dots.pop(worse_index)

        x_centre = [0] * n

        for j in range(n):
            x_centre[j] = sum([point[j] for point in dots]) / n

        x_normal = worse_x + repeat * (np.array(x_centre) - worse_x)
        x_stretched = worse_x + stretch * (np.array(x_centre) - worse_x)
        x_squeezed = worse_x + squeeze * (np.array(x_centre) - worse_x)
        simplex_base = min(x_normal, x_stretched, x_squeezed, key=lambda x: f(*x))

        func_eval += 3

        dots.insert(worse_index, simplex_base)

        if iters > max_iter:
            break

        max_distance = max(np.linalg.norm(point - best_x) for point in dots)
        if eps > max_distance:
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
    path_koshi, iters_koshi, time_koshi, evals_koshi = koshi_gradient_descent(start_p, G, G_x1, G_x2)
    path_hj, iters_hj, time_hj, evals_hj = hooke_jeeves(start_p, G, h=0.4)

    path_simplex, iters_simplex, time_simplex, evals_simplex = simplex_method(start_p, G)

    fig = plt.figure(figsize=(18, 6))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis')

    z_classic = G(path_classic[:, 0], path_classic[:, 1])
    ax1.plot(path_classic[:, 0], path_classic[:, 1], z_classic, 'r.-', label='Classic GD', zorder=10)

    z_koshi = G(path_koshi[:, 0], path_koshi[:, 1])
    ax1.plot(path_koshi[:, 0], path_koshi[:, 1], z_koshi, 'b.-', label='Koshi GD', zorder=11)

    ax1.set_title("Classic & Koshi Methods")
    ax1.legend()

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, Z, cmap='viridis')

    z_hj = G(path_hj[:, 0], path_hj[:, 1])
    ax2.plot(path_hj[:, 0], path_hj[:, 1], z_hj, 'g.-', label='Hooke-Jeeves', zorder=12)

    for i, simplex in enumerate(path_simplex):
        sx = simplex[:, 0]
        sy = simplex[:, 1]
        sz = G(sx, sy)

        label = 'Simplex Method' if i == 0 else ""

        ax2.plot(sx, sy, sz, color='magenta', linestyle='-', marker='.', label=label, zorder=13)

    ax2.set_title("Straight Methods (H-J & Simplex)")
    ax2.legend()

    ax3 = fig.add_subplot(133)
    contour = ax3.contourf(X, Y, Z, cmap='viridis', levels=40)
    plt.colorbar(contour, ax=ax3)

    ax3.plot(path_classic[:, 0], path_classic[:, 1], 'r-', label='Classic')
    ax3.plot(path_koshi[:, 0], path_koshi[:, 1], 'b-', label='Koshi')
    ax3.plot(path_hj[:, 0], path_hj[:, 1], 'g-', label='H-J')

    for i, simplex in enumerate(path_simplex):
        sx = simplex[:, 0]
        sy = simplex[:, 1]
        label = 'Simplex' if i == 0 else ""
        ax3.plot(sx, sy, color='magenta', linestyle='-', label=label)

    ax3.scatter(start_p[0], start_p[1], color='yellow', edgecolors='black', s=70, label='Start', zorder=15)

    ax3.scatter(path_classic[-1, 0], path_classic[-1, 1], color='red', marker='X', s=100, edgecolors='white',
                label='End Classic', zorder=16)
    ax3.scatter(path_koshi[-1, 0], path_koshi[-1, 1], color='blue', marker='X', s=100, edgecolors='white',
                label='End Koshi', zorder=16)
    ax3.scatter(path_hj[-1, 0], path_hj[-1, 1], color='green', marker='X', s=100, edgecolors='white', label='End H-J',
                zorder=16)

    end_simplex = path_simplex[-1][0]
    ax3.scatter(end_simplex[0], end_simplex[1], color='magenta', marker='X', s=100, edgecolors='white',
                label='End Simplex', zorder=16)

    ax3.set_xlim([-n, n])
    ax3.set_ylim([-n, n])
    ax3.set_title("Path Comparison (2D)")
    ax3.legend()

    print(path_classic[-1], path_koshi[-1], path_hj[-1], path_simplex[-1])
    steps = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    points = [(np.random.uniform(-4, 4), np.random.uniform(-4, 4)) for i in range(4)]

    for p in points:
        analytic_list = []
        numeric_list = []
        error_list = []

        grad_a = np.array([G_x1(*p), G_x2(*p)])

        for h in steps:
            grad_n = np.array(numerical_gradient(G, *p, h))
            err = np.linalg.norm(grad_a - grad_n)

            analytic_list.append(grad_a)
            numeric_list.append(grad_n)
            error_list.append(err)

        print(f"\nРезультати для точки: {p}")
        df_grad = pd.DataFrame({
            "Step (h)": steps,
            "Analytic": [np.round(g, 6) for g in analytic_list],
            "Numerical": [np.round(g, 6) for g in numeric_list],
            "Error (L2)": error_list
        })
        print(df_grad.to_string(index=False))

    summary_data = {
        "Метод": ["Класичний градієнтний", "Найшвидший спуск (Коші)", "Хука-Дживса (прямий)", "Симплекс-метод"],
        "Час виконання (с)": [time_classic, time_koshi, time_hj, time_simplex],
        "Ітерації": [iters_classic, iters_koshi, iters_hj, iters_simplex],
        "КОЦФ": [evals_classic, evals_koshi, evals_hj, evals_simplex]
    }

    df_summary = pd.DataFrame(summary_data)
    print("\n" + "=" * 65)
    print("Зведена таблиця результатів: ")
    print("=" * 65)
    print(df_summary.to_string(index=False))
    print("=" * 65 + "\n")

    plt.show()