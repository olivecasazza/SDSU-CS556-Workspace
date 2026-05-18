import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Problem 4
    **Language: Python3**

    Consider robot trajectory generation using a two-segment spline where each segment is a cubic polynomial (see equation 7.10 of the text).

    a. Sketch the graphs of position, velocity and acceleration for initial angle 0=5.0 deg, the via-point angle v=15.0 deg. and final angle f=40.0 deg. Assume that the duration of each segment is 1.0 sec (i.e. total duration of  2 sec), and the velocity at the via point is to be 17.5 deg/sec.

    b. Use the cycloid trajectory, with the same initial 0=5.0 deg and final f=40.0 deg  joint angles, and the total duration of 2 sec. Compare the maximum velocity and acceleration with the two segment cubic polynomial in (a), which trajectory is better and why?

    The maximum velocity and acceleration for the cycloid trajectory are a lot more continuous and smooth than the trajectories produced by the spline functions. This is more desireable because it will produce robot movements that are a lot smoother than those produced by the spline calculations.
    """)
    return


@app.cell
async def _():
    import micropip
    await micropip.install(['plotly', 'sympy', 'numpy'])

    import plotly.graph_objects as go
    import sympy as sp
    import numpy as np
    import math as math

    return go, math, np, sp


@app.cell
def _(go, np, sp):
    a0, a1, a2, a3, t = sp.symbols('a0 a1 a2 a3 t')
    _pf = a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3
    # position plot
    _pfl = sp.lambdify((a0, a1, a2, a3, t), _pf, 'numpy')
    _px_one = np.linspace(0, 1, 100).tolist()
    _px_two = np.linspace(1, 2, 100).tolist()
    _py_one = [_pfl(5, 0, 12.5, -2.5, t) for t in _px_one]
    _py_two = [_pfl(15, 17.5, 40, -32.5, t - 1) for t in _px_two]
    _position_plot = go.Figure()
    _position_plot.add_trace(go.Scatter(x=_px_one, y=_py_one, mode='lines'))
    _position_plot.add_trace(go.Scatter(x=_px_two, y=_py_two, mode='lines'))
    _position_plot.show()
    _vf = a1 + 2 * a2 * t + 3 * a3 * t ** 2
    _vfl = sp.lambdify((a1, a2, a3, t), _vf, 'numpy')
    _vx_one = np.linspace(0, 1, 100).tolist()
    _vx_two = np.linspace(1, 2, 100).tolist()
    # velocity plot
    _vy_one = [_vfl(0, 12.5, -2.5, t) for t in _vx_one]
    _vy_two = [_vfl(17.5, 40, -32.5, t - 1) for t in _vx_two]
    _velocity_plot = go.Figure()
    _velocity_plot.add_trace(go.Scatter(x=_vx_one, y=_vy_one, mode='lines'))
    _velocity_plot.add_trace(go.Scatter(x=_vx_two, y=_vy_two, mode='lines'))
    _velocity_plot.show()
    _af = 2 * a2 + 6 * a3 * t
    _afl = sp.lambdify((a1, a2, a3, t), _af, 'numpy')
    _ax_one = np.linspace(0, 1, 100).tolist()
    _ax_two = np.linspace(1, 2, 100).tolist()
    _ay_one = [_afl(0, 12.5, -2.5, t) for t in _ax_one]
    _ay_two = [_afl(17.5, 40, -32.5, t - 1) for t in _ax_two]
    _acceleration_plot = go.Figure()
    # acceleration plot
    _acceleration_plot.add_trace(go.Scatter(x=_ax_one, y=_ay_one, mode='lines'))
    _acceleration_plot.add_trace(go.Scatter(x=_ax_two, y=_ay_two, mode='lines'))
    _acceleration_plot.show()
    return (t,)


@app.cell
def _(go, math, np, sp, t):
    # position plot
    _pf = 5 + (t / 2 - sp.sin(2 * math.pi * t / 2) / (2 * math.pi)) * (40 - 5)
    _pfl = sp.lambdify(t, _pf, 'numpy')
    _px_one = np.linspace(0, 1, 100).tolist()
    _px_two = np.linspace(1, 2, 100).tolist()
    _py_one = [_pfl(t) for t in _px_one]
    _py_two = [_pfl(t) for t in _px_two]
    _position_plot = go.Figure()
    _position_plot.add_trace(go.Scatter(x=_px_one, y=_py_one, mode='lines'))
    _position_plot.add_trace(go.Scatter(x=_px_two, y=_py_two, mode='lines'))
    _position_plot.show()
    _vf = (1 / 2 - sp.cos(2 * math.pi * t / 2) / 2) * (40 - 5)
    _vfl = sp.lambdify(t, _vf, 'numpy')
    _vx_one = np.linspace(0, 1, 100).tolist()
    # velocity plot
    _vx_two = np.linspace(1, 2, 100).tolist()
    _vy_one = [_vfl(t) for t in _vx_one]
    _vy_two = [_vfl(t) for t in _vx_two]
    _velocity_plot = go.Figure()
    _velocity_plot.add_trace(go.Scatter(x=_vx_one, y=_vy_one, mode='lines'))
    _velocity_plot.add_trace(go.Scatter(x=_vx_two, y=_vy_two, mode='lines'))
    _velocity_plot.show()
    _af = sp.sin(2 * math.pi * t / 2) * 2 * math.pi / 4 * 35
    _afl = sp.lambdify(t, _af, 'numpy')
    _ax_one = np.linspace(0, 1, 100).tolist()
    _ax_two = np.linspace(1, 2, 100).tolist()
    _ay_one = [_afl(t) for t in _ax_one]
    _ay_two = [_afl(t) for t in _ax_two]
    # acceleration plot
    _acceleration_plot = go.Figure()
    _acceleration_plot.add_trace(go.Scatter(x=_ax_one, y=_ay_one, mode='lines'))
    _acceleration_plot.add_trace(go.Scatter(x=_ax_two, y=_ay_two, mode='lines'))
    _acceleration_plot.show()
    return


if __name__ == "__main__":
    app.run()

