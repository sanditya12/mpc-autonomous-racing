from mpc_class import MPCComponent
from simulate import simulate
import numpy as np
from casadi import sin, cos, pi
from track_data import center_sample_points


center_points = center_sample_points.center_points

obs = [
    # {"x": 4, "y": 2.5, "diameter": 4},
]

mpc = MPCComponent()
state_init = np.array([1.5, 12, 0])
state_target = np.array([6, 1.5, 0])
mpc.init_symbolic_vars()
mpc.init_cost_fn_and_g_constraints()

mpc.add_track_constraints(center_points, 3)

mpc.init_solver()
mpc.init_constraint_args()

mpc.add_track_args()

mpc.prepare_step(state_init)
mpc.init_sim_params()
while mpc.mpc_completed != True:
    u = mpc.step_with_sim_params(state_init, state_target)
    print(" ------ ")
    state_init = mpc.simulate_step_shift(u, state_init)
    # mpc.mpc_completed = True


sim_params = mpc.get_simulation_params()
simulate(
    sim_params["cat_states"],
    sim_params["cat_controls"],
    sim_params["times"],
    sim_params["step_horizon"],
    sim_params["N"],
    sim_params["p_arr"],
    sim_params["obs"],
    sim_params["rob_diam"],
    save=False,
)
