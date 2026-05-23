import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full", app_title="Kinematics")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    intro = mo.md(r"""
    # Kinematics

    This notebook compares two joint-space trajectory generators for a single revolute joint:

    1. a two-segment cubic polynomial that passes through a via point, and
    2. a cycloid trajectory between the same start and finish angles.

    Move the controls and the plots recompute in-browser. The goal is to see how the via-point velocity and segment duration change position, velocity, and acceleration continuity.
    """)
    return (intro,)


@app.cell(hide_code=True)
async def _():
    import micropip

    await micropip.install(["plotly", "numpy"])

    import math
    import numpy as np
    import plotly.graph_objects as go

    return go, math, np


@app.cell(hide_code=True)
def _(mo):
    theta_start = mo.ui.slider(0, 30, value=5, step=1, label="initial angle θ₀ (deg)")
    theta_via = mo.ui.slider(5, 45, value=15, step=1, label="via-point angle θᵥ (deg)")
    theta_final = mo.ui.slider(20, 80, value=40, step=1, label="final angle θf (deg)")
    segment_duration = mo.ui.slider(0.5, 3.0, value=1.0, step=0.25, label="segment duration (sec)")
    via_velocity = mo.ui.slider(-20, 50, value=17.5, step=0.5, label="via-point velocity (deg/sec)")

    controls = mo.vstack([
        mo.md("## Parameters"),
        mo.hstack([theta_start, theta_via, theta_final], justify="start"),
        mo.hstack([segment_duration, via_velocity], justify="start"),
    ])
    mo.output.append(controls)
    return controls, segment_duration, theta_final, theta_start, theta_via, via_velocity


@app.cell(hide_code=True)
def _(np, segment_duration, theta_final, theta_start, theta_via, via_velocity):
    q0 = float(theta_start.value)
    qv = float(theta_via.value)
    qf = float(theta_final.value)
    T = float(segment_duration.value)
    vv = float(via_velocity.value)

    def cubic_coefficients(q_start, q_end, v_start, v_end, duration):
        a0 = q_start
        a1 = v_start
        a2 = (3 * (q_end - q_start) / duration**2) - ((2 * v_start + v_end) / duration)
        a3 = (-2 * (q_end - q_start) / duration**3) + ((v_start + v_end) / duration**2)
        return a0, a1, a2, a3

    first_segment = cubic_coefficients(q0, qv, 0.0, vv, T)
    second_segment = cubic_coefficients(qv, qf, vv, 0.0, T)

    sample_count = 180
    t_first = np.linspace(0, T, sample_count)
    t_second_local = np.linspace(0, T, sample_count)
    t_second = t_second_local + T
    t_all = np.concatenate([t_first, t_second])

    def cubic_values(coefficients, t_values):
        a0, a1, a2, a3 = coefficients
        position = a0 + a1 * t_values + a2 * t_values**2 + a3 * t_values**3
        velocity = a1 + 2 * a2 * t_values + 3 * a3 * t_values**2
        acceleration = 2 * a2 + 6 * a3 * t_values
        return position, velocity, acceleration

    p1, v1, a1 = cubic_values(first_segment, t_first)
    p2, v2, a2 = cubic_values(second_segment, t_second_local)
    cubic_position = np.concatenate([p1, p2])
    cubic_velocity = np.concatenate([v1, v2])
    cubic_acceleration = np.concatenate([a1, a2])

    total_duration = 2 * T
    t_cycloid = np.linspace(0, total_duration, sample_count * 2)
    delta = qf - q0
    cycloid_position = q0 + ((t_cycloid / total_duration) - np.sin(2 * np.pi * t_cycloid / total_duration) / (2 * np.pi)) * delta
    cycloid_velocity = ((1 / total_duration) - np.cos(2 * np.pi * t_cycloid / total_duration) / total_duration) * delta
    cycloid_acceleration = (np.sin(2 * np.pi * t_cycloid / total_duration) * 2 * np.pi / total_duration**2) * delta

    trajectory_data = {
        "q0": q0,
        "qv": qv,
        "qf": qf,
        "T": T,
        "vv": vv,
        "t_all": t_all,
        "t_cycloid": t_cycloid,
        "cubic_position": cubic_position,
        "cubic_velocity": cubic_velocity,
        "cubic_acceleration": cubic_acceleration,
        "cycloid_position": cycloid_position,
        "cycloid_velocity": cycloid_velocity,
        "cycloid_acceleration": cycloid_acceleration,
        "first_segment": first_segment,
        "second_segment": second_segment,
    }
    return (trajectory_data,)


@app.cell(hide_code=True)
def _(go):
    def dark_figure(title, y_title):
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            title=title,
            xaxis_title="time (sec)",
            yaxis_title=y_title,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=40, r=20, t=70, b=40),
        )
        return fig

    return (dark_figure,)


@app.cell(hide_code=True)
def _(dark_figure, go, mo, trajectory_data):
    position_plot = dark_figure("Position", "joint angle (deg)")
    position_plot.add_trace(go.Scatter(x=trajectory_data["t_all"], y=trajectory_data["cubic_position"], mode="lines", name="two-segment cubic"))
    position_plot.add_trace(go.Scatter(x=trajectory_data["t_cycloid"], y=trajectory_data["cycloid_position"], mode="lines", name="cycloid"))
    position_plot.add_vline(x=trajectory_data["T"], line_dash="dot", line_color="#9ca3af")

    velocity_plot = dark_figure("Velocity", "joint velocity (deg/sec)")
    velocity_plot.add_trace(go.Scatter(x=trajectory_data["t_all"], y=trajectory_data["cubic_velocity"], mode="lines", name="two-segment cubic"))
    velocity_plot.add_trace(go.Scatter(x=trajectory_data["t_cycloid"], y=trajectory_data["cycloid_velocity"], mode="lines", name="cycloid"))
    velocity_plot.add_vline(x=trajectory_data["T"], line_dash="dot", line_color="#9ca3af")

    acceleration_plot = dark_figure("Acceleration", "joint acceleration (deg/sec²)")
    acceleration_plot.add_trace(go.Scatter(x=trajectory_data["t_all"], y=trajectory_data["cubic_acceleration"], mode="lines", name="two-segment cubic"))
    acceleration_plot.add_trace(go.Scatter(x=trajectory_data["t_cycloid"], y=trajectory_data["cycloid_acceleration"], mode="lines", name="cycloid"))
    acceleration_plot.add_vline(x=trajectory_data["T"], line_dash="dot", line_color="#9ca3af")

    plots = mo.vstack([
        mo.md("## Trajectory comparison"),
        mo.md("The dotted vertical line marks the via point between cubic segments."),
        position_plot,
        velocity_plot,
        acceleration_plot,
    ])
    mo.output.append(plots)
    return acceleration_plot, plots, position_plot, velocity_plot


@app.cell(hide_code=True)
def _(mo, np, trajectory_data):
    max_cubic_velocity = float(np.max(np.abs(trajectory_data["cubic_velocity"])))
    max_cubic_acceleration = float(np.max(np.abs(trajectory_data["cubic_acceleration"])))
    max_cycloid_velocity = float(np.max(np.abs(trajectory_data["cycloid_velocity"])))
    max_cycloid_acceleration = float(np.max(np.abs(trajectory_data["cycloid_acceleration"])))

    comparison = mo.md(f"""
    ## What changes?

    | trajectory | max velocity | max acceleration |
    | --- | ---: | ---: |
    | two-segment cubic | `{max_cubic_velocity:.2f}` deg/sec | `{max_cubic_acceleration:.2f}` deg/sec² |
    | cycloid | `{max_cycloid_velocity:.2f}` deg/sec | `{max_cycloid_acceleration:.2f}` deg/sec² |

    The cubic trajectory is useful when a via-point and via-point velocity are required, but acceleration can jump at the segment boundary. The cycloid trajectory is smoother between the start and final angle because velocity and acceleration vary continuously over the full motion.
    """)
    mo.output.append(comparison)
    return (comparison,)


@app.cell(hide_code=True)
def _(mo, trajectory_data):
    first = trajectory_data["first_segment"]
    second = trajectory_data["second_segment"]
    equations = mo.md(f"""
    ## Cubic coefficients

    First segment coefficients `(a0, a1, a2, a3)`:
    ```text
    ({first[0]:.3f}, {first[1]:.3f}, {first[2]:.3f}, {first[3]:.3f})
    ```

    Second segment coefficients `(a0, a1, a2, a3)`:
    ```text
    ({second[0]:.3f}, {second[1]:.3f}, {second[2]:.3f}, {second[3]:.3f})
    ```
    """)
    mo.output.append(equations)
    return (equations,)


if __name__ == "__main__":
    app.run()
