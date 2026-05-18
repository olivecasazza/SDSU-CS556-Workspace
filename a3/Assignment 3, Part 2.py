import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full", app_title="Inverse kinematics approximation")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Inverse kinematics approximation

    A lightweight interactive version of the original 2R robot-arm homework. The
    exact inverse kinematics solution is compared with a local linear
    approximation trained on nearby forward-kinematics samples.

    The original notebook ran tens of thousands of samples and KMeans clusters
    in-browser. This version keeps the math visible but uses a small local fit so
    it is responsive in Marimo WASM.
    """)
    return


@app.cell
async def _():
    import micropip

    await micropip.install(["numpy", "plotly", "scikit-learn"])

    import math
    import numpy as np
    import plotly.graph_objects as go
    from sklearn.linear_model import LinearRegression

    return LinearRegression, go, math, np


@app.cell
def _(mo):
    l1 = mo.ui.slider(40, 160, value=100, step=5, label="link 1 length")
    l2 = mo.ui.slider(40, 160, value=100, step=5, label="link 2 length")
    target_x = mo.ui.slider(-200, 200, value=85, step=5, label="target x")
    target_y = mo.ui.slider(-200, 200, value=105, step=5, label="target y")
    neighborhood = mo.ui.slider(5, 40, value=20, step=5, label="local samples per dimension")
    controls = mo.vstack([l1, l2, target_x, target_y, neighborhood])
    controls
    return l1, l2, neighborhood, target_x, target_y


@app.cell
def _(math, np):
    def forward_kinematics(theta1, theta2, l1, l2):
        return np.array([
            l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2),
            l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2),
        ])

    def exact_inverse_kinematics(x, y, l1, l2):
        r2 = x**2 + y**2
        cos_theta2 = (r2 - l1**2 - l2**2) / (2 * l1 * l2)
        if cos_theta2 < -1 or cos_theta2 > 1:
            return []

        theta2_options = [math.acos(cos_theta2), -math.acos(cos_theta2)]
        solutions = []
        for theta2 in theta2_options:
            theta1 = math.atan2(y, x) - math.atan2(
                l2 * math.sin(theta2),
                l1 + l2 * math.cos(theta2),
            )
            solutions.append(np.array([theta1, theta2]))
        return solutions

    return exact_inverse_kinematics, forward_kinematics


@app.cell
def _(
    LinearRegression,
    exact_inverse_kinematics,
    forward_kinematics,
    l1,
    l2,
    neighborhood,
    np,
    target_x,
    target_y,
):
    link1 = float(l1.value)
    link2 = float(l2.value)
    target = np.array([float(target_x.value), float(target_y.value)])
    exact_solutions = exact_inverse_kinematics(target[0], target[1], link1, link2)

    # Fit around the elbow-down branch when reachable. If the target is outside
    # the workspace, use the nearest stretched configuration so the visualization
    # still explains why no exact IK solution exists.
    if exact_solutions:
        center = exact_solutions[0]
    else:
        center = np.array([np.atan2(target[1], target[0]), 0.0])

    span = np.deg2rad(18)
    n = int(neighborhood.value)
    theta1_grid = np.linspace(center[0] - span, center[0] + span, n)
    theta2_grid = np.linspace(center[1] - span, center[1] + span, n)

    angles = []
    points = []
    for theta1 in theta1_grid:
        for theta2 in theta2_grid:
            angles.append([theta1, theta2])
            points.append(forward_kinematics(theta1, theta2, link1, link2))

    angles = np.array(angles)
    points = np.array(points)
    model = LinearRegression().fit(points, angles)
    predicted_angles = model.predict([target])[0]
    predicted_point = forward_kinematics(predicted_angles[0], predicted_angles[1], link1, link2)
    error = float(np.linalg.norm(predicted_point - target))

    return angles, error, exact_solutions, link1, link2, points, predicted_angles, predicted_point, target


@app.cell
def _(exact_solutions, forward_kinematics, go, link1, link2, np, points, predicted_point, target):
    reach = link1 + link2
    theta = np.linspace(0, 2 * np.pi, 240)
    workspace_x = reach * np.cos(theta)
    workspace_y = reach * np.sin(theta)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=workspace_x, y=workspace_y, mode="lines", name="workspace boundary", line=dict(color="#4b5563")))
    fig.add_trace(go.Scattergl(x=points[:, 0], y=points[:, 1], mode="markers", name="local training samples", marker=dict(size=4, color="#5dcdbe", opacity=0.45)))
    fig.add_trace(go.Scatter(x=[target[0]], y=[target[1]], mode="markers", name="target", marker=dict(size=12, color="#f0dd7d", symbol="x")))
    fig.add_trace(go.Scatter(x=[predicted_point[0]], y=[predicted_point[1]], mode="markers", name="linear approximation", marker=dict(size=11, color="#eba798")))

    for idx, solution in enumerate(exact_solutions):
        p1 = forward_kinematics(solution[0], 0, link1, 0)
        p2 = forward_kinematics(solution[0], solution[1], link1, link2)
        fig.add_trace(go.Scatter(x=[0, p1[0], p2[0]], y=[0, p1[1], p2[1]], mode="lines+markers", name=f"exact arm {idx + 1}", line=dict(width=4)))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=640,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(scaleanchor="y", scaleratio=1, title="x"),
        yaxis=dict(title="y"),
        legend=dict(orientation="h"),
    )
    fig
    return


@app.cell(hide_code=True)
def _(error, exact_solutions, mo, np, predicted_angles):
    reachable = "yes" if exact_solutions else "no"
    mo.md(
        f"""
        ## Approximation result

        | quantity | value |
        | --- | ---: |
        | target reachable | {reachable} |
        | predicted θ₁ | {np.rad2deg(predicted_angles[0]):.2f}° |
        | predicted θ₂ | {np.rad2deg(predicted_angles[1]):.2f}° |
        | approximation error | {error:.2f} units |

        A linear approximation works well inside a small neighborhood of a
        reachable target. Error grows near singularities, workspace boundaries,
        and branch switches where the inverse kinematics mapping stops behaving
        like a single smooth plane.
        """
    )
    return


if __name__ == "__main__":
    app.run()
