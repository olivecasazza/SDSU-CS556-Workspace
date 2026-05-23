import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full", app_title="Inverse Kinematics Approximation")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    intro = mo.md(r"""
    # Inverse Kinematics Approximation

    This notebook explores finding the inverse kinematics of a 2PR manipulator arm using decomposition and linear approximation. Instead of solving the exact inverse kinematics globally, we:

    1. Discretize the workspace into a grid.
    2. Generate thousands of random joint angles (forward kinematics).
    3. Use K-Means clustering inside each cell to split the two possible elbow configurations.
    4. Fit a local linear regression to approximate the inverse kinematics in that region.

    Below, you can adjust the arm parameters and approximation density.
    """)
    return (intro,)


@app.cell(hide_code=True)
async def _():
    import micropip

    await micropip.install([
        "matplotlib",
        "numpy",
        "sympy",
        "pandas",
        "plotly",
        "scikit-learn",
    ])

    import math
    import random
    import numpy as np
    import sympy as sp
    import pandas as pd
    from sympy.physics.mechanics import dynamicsymbols
    import plotly.graph_objects as go
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LinearRegression

    return (
        KMeans,
        LinearRegression,
        dynamicsymbols,
        go,
        math,
        np,
        pd,
        random,
        sp,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.output.append(mo.image(src="fig/img1.png", width=300))
    return


@app.cell(hide_code=True)
def _(mo):
    l1_len = mo.ui.slider(50, 150, value=100, step=10, label="link 1 length (cm)")
    l2_len = mo.ui.slider(50, 150, value=100, step=10, label="link 2 length (cm)")
    trials = mo.ui.slider(5000, 50000, value=10000, step=5000, label="random samples")
    grid_size = mo.ui.slider(10, 100, value=40, step=5, label="grid dimension (N x N)")

    controls = mo.vstack([
        mo.md("## Arm and simulation parameters"),
        mo.hstack([l1_len, l2_len], justify="start"),
        mo.hstack([trials, grid_size], justify="start"),
    ])
    mo.output.append(controls)
    return controls, grid_size, l1_len, l2_len, trials


@app.cell(hide_code=True)
def _(dynamicsymbols, mo, sp):
    theta1, theta2, l1, l2, theta, alpha, a, d = dynamicsymbols(
        "theta1 theta2 l1 l2 theta alpha a d"
    )

    rotation_matrix = sp.Matrix(
        [
            [sp.cos(theta), -sp.sin(theta) * sp.cos(alpha), sp.sin(theta) * sp.sin(alpha)],
            [sp.sin(theta), sp.cos(theta) * sp.cos(alpha), -sp.cos(theta) * sp.sin(alpha)],
            [0, sp.sin(alpha), sp.cos(alpha)],
        ]
    )

    transformation_matrix = sp.Matrix([a * sp.cos(theta), a * sp.sin(theta), d])
    last_row = sp.Matrix([[0, 0, 0, 1]])

    t = sp.Matrix.vstack(sp.Matrix.hstack(rotation_matrix, transformation_matrix), last_row)
    t_02 = t.subs({alpha: 0, a: l1, theta: theta1, d: 0}) * t.subs({alpha: 0, a: l2, theta: theta2, d: 0})
    t_02.simplify()

    derivation = mo.vstack([
        mo.md("## Forward kinematics transformation matrix"),
        mo.md("Using the DH parameters, we derive the exact forward kinematics for the end-effector:"),
        mo.md(f"$$\n{sp.latex(t_02)}\n$$"),
    ])
    mo.output.append(derivation)
    return a, alpha, d, derivation, l1, l2, last_row, rotation_matrix, t, t_02, theta, theta1, theta2, transformation_matrix


@app.cell(hide_code=True)
def _(l1, l1_len, l2, l2_len, math, sp, t_02, theta1, theta2):
    x_position = t_02[0, 3]
    y_position = t_02[1, 3]
    fk_x = sp.lambdify((l1, l2, theta1, theta2), x_position, "numpy")
    fk_y = sp.lambdify((l1, l2, theta1, theta2), y_position, "numpy")

    def fk(t1, t2):
        return (
            fk_x(l1_len.value, l2_len.value, t1, t2),
            fk_y(l1_len.value, l2_len.value, t1, t2),
        )

    def exact_ik(x, y):
        try:
            L1 = l1_len.value
            L2 = l2_len.value
            a = x**2 + y**2 - L1**2 - L2**2
            b = 2 * L1 * L2
            t2_pos = math.acos(a / b)
            t2_neg = -1 * t2_pos

            t1_pos = math.atan2(y, x) - math.atan2(L1 * math.sin(t2_pos), L1 + L2 * math.cos(t2_pos))
            t1_neg = math.atan2(y, x) - math.atan2(L1 * math.sin(t2_neg), L1 + L2 * math.cos(t2_neg))
            return [(t1_pos, t2_pos), (t1_neg, t2_neg)]
        except ValueError:
            return None

    return exact_ik, fk, fk_x, fk_y, x_position, y_position


@app.cell(hide_code=True)
def _(fk, go, grid_size, math, mo, np, random, trials):
    trial_data = []
    for _ in range(trials.value):
        t1 = np.deg2rad(random.uniform(0, 170))
        t2 = np.deg2rad(random.uniform(-90, 90))
        trial_data.append({
            "angles": (t1, t2),
            "pos": fk(t1, t2)
        })

    x_cords = [d["pos"][0] for d in trial_data]
    y_cords = [d["pos"][1] for d in trial_data]

    fig_fuzzed = go.Figure(data=go.Scattergl(x=x_cords, y=y_cords, mode='markers', marker=dict(size=2)))
    fig_fuzzed.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=0))

    x_min, y_min = math.floor(min(x_cords)), math.floor(min(y_cords))
    x_max, y_max = math.ceil(max(x_cords)), math.ceil(max(y_cords))
    w_width, w_height = x_max - x_min, y_max - y_min

    cols = grid_size.value
    rows = grid_size.value
    cell_w = w_width / cols
    cell_h = w_height / rows
    gw = math.ceil(w_width / cell_w)
    gh = math.ceil(w_height / cell_h)

    def get_grid(x, y):
        _x = math.floor(((x - x_min) / w_width) * gw)
        _y = math.floor(((y - y_min) / w_height) * gh)
        return _x, _y

    fuzz_section = mo.vstack([
        mo.md("## Fuzzed workspace"),
        mo.md("Random joint angles mapped through forward kinematics to show the reachable workspace:"),
        fig_fuzzed,
    ])
    mo.output.append(fuzz_section)
    return (
        cell_h,
        cell_w,
        cols,
        fig_fuzzed,
        fuzz_section,
        get_grid,
        gh,
        gw,
        rows,
        trial_data,
        w_height,
        w_width,
        x_cords,
        x_max,
        x_min,
        y_cords,
        y_max,
        y_min,
    )


@app.cell(hide_code=True)
def _(KMeans, LinearRegression, get_grid, gh, gw, pd, trial_data):
    grid = [[[] for _ in range(gw)] for _ in range(gh)]
    for pt in trial_data:
        _gx, _gy = get_grid(pt["pos"][0], pt["pos"][1])
        if 0 <= _gx < gw and 0 <= _gy < gh:
            grid[_gy][_gx].append(pt)

    def split_and_fit(cell):
        if not cell or len(cell) < 2:
            return None
        t_pairs = [pt["angles"] for pt in cell]
        xy_pairs = [pt["pos"] for pt in cell]
        kmeans = KMeans(n_clusters=2, n_init=1)
        clusters = kmeans.fit_predict(t_pairs)
        
        c0_t = [t_pairs[i] for i, c in enumerate(clusters) if c == 0]
        c0_xy = [xy_pairs[i] for i, c in enumerate(clusters) if c == 0]
        c1_t = [t_pairs[i] for i, c in enumerate(clusters) if c == 1]
        c1_xy = [xy_pairs[i] for i, c in enumerate(clusters) if c == 1]
        
        models = []
        if c0_t:
            models.append(LinearRegression().fit(c0_xy, c0_t))
        if c1_t:
            models.append(LinearRegression().fit(c1_xy, c1_t))
        return models if len(models) == 2 else None

    grid_df = pd.DataFrame(grid)
    model_grid = grid_df.map(split_and_fit)
    return grid, grid_df, model_grid, split_and_fit


@app.cell(hide_code=True)
def _(
    exact_ik,
    fk,
    get_grid,
    go,
    math,
    mo,
    model_grid,
    random,
    x_max,
    x_min,
    y_max,
    y_min,
):
    test_inputs = []
    trial_errors = []
    
    attempts = 0
    while len(trial_errors) < 1000 and attempts < 10000:
        attempts += 1
        px, py = random.uniform(x_min, x_max), random.uniform(y_min, y_max)
        _gx, _gy = get_grid(px, py)
        ik_res = exact_ik(px, py)
        
        try:
            models = model_grid[_gx][_gy]
        except (KeyError, IndexError):
            continue
            
        if models is None or ik_res is None:
            continue
            
        test_inputs.append((px, py))
        m1, m2 = models
        sol1 = m1.predict([[px, py]])[-1]
        sol2 = m2.predict([[px, py]])[-1]
        
        fk1 = fk(*sol1)
        fk2 = fk(*sol2)
        
        err1 = abs(math.hypot(px - fk1[0], py - fk1[1]))
        err2 = abs(math.hypot(px - fk2[0], py - fk2[1]))
        
        trial_errors.append(min(err1, err2))

    avg_error = sum(trial_errors) / len(trial_errors) if trial_errors else 0

    x_test = [p[0] for p in test_inputs]
    y_test = [p[1] for p in test_inputs]
    size_test = [e + 3 for e in trial_errors]

    fig_err = go.Figure(data=go.Scattergl(
        x=x_test, y=y_test, mode='markers',
        marker=dict(size=size_test, color=trial_errors, colorscale='Viridis', showscale=True)
    ))
    fig_err.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=0))

    fig_err_3d = go.Figure(data=go.Scatter3d(
        x=x_test, y=y_test, z=trial_errors, mode='markers',
        marker=dict(size=2, color=trial_errors, colorscale='Viridis', showscale=True)
    ))
    fig_err_3d.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(xaxis_title='X Position', yaxis_title='Y Position', zaxis_title='Error (cm)')
    )

    err_section = mo.vstack([
        mo.md(f"## Approximation Error\nAverage error over 1000 samples: `{avg_error:.3f}` cm"),
        mo.md("The 2D and 3D scatter plots below highlight regions where the local linear approximation struggles (higher Z-axis / lighter color means higher error, usually around workspace boundaries and singularities)."),
        mo.hstack([fig_err, fig_err_3d])
    ])
    mo.output.append(err_section)

    return attempts, avg_error, err_section, fig_err, fig_err_3d, size_test, test_inputs, trial_errors, x_test, y_test


if __name__ == "__main__":
    app.run()
