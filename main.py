from mpc_class_loop import MPCComponent
from simulate import simulate
import numpy as np
from casadi import sin, cos, pi
from track_data import center_sample_points
from time import time
from reference_generator import ReferenceGenerator

center_points = center_sample_points.center_points

x_init = 1.5
y_init = 5
theta_init = 0
horizon = 7
mpc = MPCComponent(vision_horizon= horizon, N = 15)

state_init = np.array([x_init, y_init, theta_init])
mpc.init_symbolic_vars()
mpc.init_cost_fn_and_g_constraints()


start_time = time()
lane_width = 3
mpc.add_track_constraints(lane_width/2)

mpc.init_solver()

mpc.init_constraint_args()

mpc.add_track_args()

mpc.prepare_step(state_init)
mpc.init_sim_params()

print("preparing time: ", time() - start_time)
#Target [1.1, 12.149999999999999]
while time() - start_time < 10:
    init_time = time()

    rg = ReferenceGenerator(horizon, center_points)
    visible_center_points = rg.generate_map((state_init[0], state_init[1]))
    x_ref, y_ref = visible_center_points[-1]
    visible_center_points = np.array(visible_center_points).flatten()
    state_target = np.array([x_ref, y_ref, 0])


    u = mpc.step_with_sim_params(state_init, state_target, visible_center_points)
    state_init = mpc.simulate_step_shift(u, state_init)

    # state_arr = np.array(state_init)
    print("Time per step: ", time() - init_time)
    # print("x: ", state_arr[0], ", y: ", state_arr[1], ", th: ", state_arr[2])
    print("  ")

print("Whole MPC time: ", time() - start_time)

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
