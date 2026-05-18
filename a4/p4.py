import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full", app_title="Robot trajectory generation")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Robot trajectory generation

    Compare a two-segment cubic spline trajectory against a cycloidal trajectory
    for a single robot joint. The cubic trajectory passes through a via point;
    the cycloid moves directly from start to finish with continuous velocity and
    acceleration at the endpoints.
    """)
    return


@app.cell
async def _():
    import micropip

    await micropip.install(["numpy", "plotly"])

    import numpy as np
    import plotly.graph_objects as go

    return go, np


@app.cell
def _(mo):
    start = mo.ui.slider(0, 90, value=5, step=1, label="start angle θ₀ (deg)")
    via = mo.ui.slider(0, 90, value=15, step=1, label="via angle θᵥ (deg)")
    finish = mo.ui.slider(0, 90, value=40, step=1, label="finish angle θf (deg)")
    via_velocity = mo.ui.slider(-60, 60, value=17.5, step=0.5, label="via velocity (deg/s)")
    duration = mo.ui.slider(0.5, 5.0, value=2.0, step=0.25, label="total duration (s)")
    controls = mo.vstack([start, via, finish, via_velocity, duration])
    controls
    return duration, finish, start, via, via_velocity


@app.cell
def _(duration, finish, go, np, start, via, via_velocity):
    q0 = float(start.value)
    qv = float(via.value)
    qf = float(finish.value)
    vv = float(via_velocity.value)
    tf = float(duration.value)
    tv = tf / 2

    t1 = np.linspace(0, tv, 160)
    t2 = np.linspace(tv, tf, 160)
    tau1 = t1
    tau2 = t2 - tv

    def cubic_coeff(q_start, q_end, v_start, v_end, segment_duration):
        a0 = q_start
        a1 = v_start
        a2 = (3 * (q_end - q_start) / segment_duration**2) - ((2 * v_start + v_end) / segment_duration)
        a3 = (-2 * (q_end - q_start) / segment_duration**3) + ((v_start + v_end) / segment_duration**2)
        return a0, a1, a2, a3

    c1 = cubic_coeff(q0, qv, 0.0, vv, tv)
    c2 = cubic_coeff(qv, qf, vv, 0.0, tv)

    def eval_cubic(coeffs, tau):
        a0, a1, a2, a3 = coeffs
        q = a0 + a1 * tau + a2 * tau**2 + a3 * tau**3
        v = a1 + 2 * a2 * tau + 3 * a3 * tau**2
        a = 2 * a2 + 6 * a3 * tau
        return q, v, a

    q1, v1, a1 = eval_cubic(c1, tau1)
    q2, v2, a2 = eval_cubic(c2, tau2)
    cubic_t = np.concatenate([t1, t2])
    cubic_q = np.concatenate([q1, q2])
    cubic_v = np.concatenate([v1, v2])
    cubic_a = np.concatenate([a1, a2])

    tc = np.linspace(0, tf, 320)
    s = tc / tf
    cycloid_q = q0 + (qf - q0) * (s - np.sin(2 * np.pi * s) / (2 * np.pi))
    cycloid_v = (qf - q0) / tf * (1 - np.cos(2 * np.pi * s))
    cycloid_a = (qf - q0) * (2 * np.pi / tf**2) * np.sin(2 * np.pi * s)

    fig = go.Figure()
    rows = [
        ("position", cubic_q, cycloid_q, "deg"),
        ("velocity", cubic_v, cycloid_v, "deg/s"),
        ("acceleration", cubic_a, cycloid_a, "deg/s²"),
    ]
    for idx, (name, cubic_y, cycloid_y, units) in enumerate(rows):
        yaxis = "y" if idx == 0 else f"y{idx + 1}"
        fig.add_trace(go.Scatter(x=cubic_t, y=cubic_y, name=f"cubic {name}", yaxis=yaxis, line=dict(color="#5dcdbe")))
        fig.add_trace(go.Scatter(x=tc, y=cycloid_y, name=f"cycloid {name}", yaxis=yaxis, line=dict(color="#f0dd7d", dash="dash")))
        fig.update_layout({
            yaxis: dict(title=f"{name} ({units})", domain=[1 - (idx + 1) / 3 + 0.04, 1 - idx / 3 - 0.04])
        })

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=760,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(title="time (s)"),
        legend=dict(orientation="h"),
    )
    fig
    return cubic_a, cubic_v, cycloid_a, cycloid_v, mo if False else None


@app.cell(hide_code=True)
def _(cubic_a, cubic_v, cycloid_a, cycloid_v, mo, np):
    mo.md(
        f"""
        ## Comparison

        | trajectory | max velocity | max acceleration |
        | --- | ---: | ---: |
        | two-segment cubic | {np.max(np.abs(cubic_v)):.2f} deg/s | {np.max(np.abs(cubic_a)):.2f} deg/s² |
        | cycloid | {np.max(np.abs(cycloid_v)):.2f} deg/s | {np.max(np.abs(cycloid_a)):.2f} deg/s² |

        The cycloid is usually preferable when smooth endpoint behavior matters:
        it starts and ends with zero velocity and zero acceleration. The cubic
        spline can satisfy the via-point constraint, but the acceleration jumps
        at segment boundaries unless extra continuity constraints are added.
        """
    )
    return


if __name__ == "__main__":
    app.run()
