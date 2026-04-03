import time
import numpy as np
import pandas as pd
from math import sin, sqrt
import matplotlib.pyplot as plt

G = lambda x: sin(x)**4 + 6 * (x - 1)**2 + 10

def bruteforce_search(f, a, b, epsilon):
    n = 0
    min = f(a)
    while a <= b:
        new = f(a)
        if new < min:
            min = new
        a += epsilon
        n += 1
    return min, n

def DSK(f, x0, h):
    n = 0

    f_mid = f(x0)
    f_right = f(x0 + h)
    f_left = f(x0 - h)
    n += 3

    if f_left >= f_mid <= f_right:
        return x0 - h, x0 + h, n

    if f_left > f_mid > f_right:
        direction = 1
        x_points = [x0, x0 + h]
        f_values = [f_mid, f_right]
    else:
        direction = -1
        x_points = [x0, x0 - h]
        f_values = [f_mid, f_left]

    k = 2
    while True:
        next_step = direction * h * (2 ** (k - 1))
        next_x = x_points[-1] + next_step
        next_f = f(next_x)
        n += 1

        if next_f > f_values[-1]:
            x_points.append(next_x)
            f_values.append(next_f)
            break
        else:
            x_points.append(next_x)
            f_values.append(next_f)
            k += 1

    l = min(x_points[-3], x_points[-1])
    r = max(x_points[-3], x_points[-1])

    return l, r, n

def dichotomy_search(f, a, b, epsilon):
    n = 1
    l = a
    r = b
    xm = (l + r) / 2
    fm = f(xm)
    x1 = (l + xm) / 2
    x2 = (xm + r) / 2
    while abs(r - l) > epsilon:
        x1 = (l + xm) / 2
        x2 = (xm + r) / 2
        f1 = f(x1)
        f2 = f(x2)
        if fm < f1 and fm < f2:
            l = x1
            r = x2
        elif fm > f1:
            r = xm
            xm = x1
            fm = f1
        else:
            l = xm
            xm = x2
            fm = f2
        n += 2
    return f((r + l) / 2), n

def golden_search(f, a, b, epsilon):
    res_phi = (sqrt(5) - 1) / 2

    l, r = a, b
    n = 2

    x1 = r - (res_phi) * (r - l)
    x2 = l + res_phi * (r - l)
    f1, f2 = f(x1), f(x2)

    while (r - l) > epsilon:
        if f1 > f2:
            l = x1
            x1, f1 = x2, f2
            x2 = l + res_phi * (r - l)
            f2 = f(x2)
        else:
            r = x2
            x2, f2 = x1, f1
            x1 = l + (1 - res_phi) * (r - l)
            f1 = f(x1)
        n += 1

    return (l + r) / 2, n

def ternary_search(f, a, b, epsilon):
    n = 0
    l = a
    r = b
    while abs(r - l) > epsilon:
        m1 = l + (r - l) / 3
        m2 = r - (r - l) / 3
        if f(m1) > f(m2):
            l = m1
        else:
            r = m2
        n += 2
    return f((r + l) / 2), n


def print_data(f, searches, intervals, eps_values, repeats=1000):
    for eps in eps_values:
        print(f"\n{'#' * 70}")
        print(f" TESTING FOR TARGET PRECISION (EPSILON): {eps}")
        print(f"{'#' * 70}")

        for inter in intervals:
            l_bound, r_bound, dsk_calls, step = inter

            print(f"\n--- Interval found with DSK (Step: {step}) ---")
            print(f"Bounds: [{l_bound:.5f}, {r_bound:.5f}] | DSK Function Calls: {dsk_calls}")

            comparison_results = []

            for search in searches:
                min_val, n_calls = search(f, l_bound, r_bound, eps)

                start_time = time.perf_counter()
                for _ in range(repeats):
                    search(f, l_bound, r_bound, eps)
                end_time = time.perf_counter()

                avg_time_ms = ((end_time - start_time) / repeats) * 1000

                comparison_results.append({
                    "Method": search.__name__.replace("_search", "").capitalize(),
                    "Result (min)": f"{min_val:.8f}",
                    "Total Calls": n_calls,
                    "Avg Time (ms)": f"{avg_time_ms:.6f}"
                })

            df = pd.DataFrame(comparison_results)
            print(df.to_string(index=False))
    pass


if __name__ == '__main__':

    a, b = 0, 2

    steps = [1, 0.1, 0.01, 0.001, 0.0001]

    intervals = []

    for step in steps:
        intervals.append([*DSK(G, a, step), step])

    epsilons = [0.1, 0.01, 0.001]

    searches = [bruteforce_search, ternary_search, dichotomy_search, golden_search]

    print_data(G, searches, intervals, epsilons)

    x = np.linspace(a, b, 1000)
    y = [G(xi) for xi in x]

    plt.plot(x, y)
    plt.grid(True)
    plt.show()
