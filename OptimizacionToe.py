import math
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize, fsolve, least_squares
from scipy.interpolate import make_interp_spline, BSpline, interp1d
import csv


class ToeOptimizer():

    def __init__(self):
        self.upper_wishbone = {
            'upright': np.array([0.0, 537.77, 290], dtype=float),
            'chassis_fore': np.array([150, 195, 256.83], dtype=float),
            'chassis_rear': np.array([-150, 195, 267.77], dtype=float)
        }
        self.lower_wishbone = {
            'upright': np.array([0.0, 585, 112.65], dtype=float),
            'chassis_fore': np.array([150, 195, 116.52], dtype=float),
            'chassis_rear': np.array([-150, 195, 100.44], dtype=float)
        }
        self.tie_rod = {

            'inner': np.array([0, 0, 0], dtype=float),
            'outer': np.array([0, 0, 0], dtype=float)
        }

        self.minimum_heave = -30
        self.maximum_heave = 30
        self.step = 1

        heave_range = range(self.minimum_heave,
                            self.maximum_heave + 1, self.step)

        for heave in heave_range:

            upper, lower, _, w_success = self.solve_by_angles(heave)
            tie_rod_out, t_success = self.solve_tie_rod(upper, lower)
            if not w_success or not t_success:
                return 1e6

            bump_steer = self.get_bump_steer(
                upper, lower, tie_rod_out, static_toe_angle=0.0)
            print(bump_steer)

        best_result = self.toe_optimizer()
        for i in range(100):
            init = best_result.x + np.random.normal(scale=100.0, size=6)
            res = self.toe_optimizer(init)

            if res.fun < best_result.fun:
                best_result = res

        print(
            f"New Inner Tie Rod: {np.round(best_result.x[0])}, {np.round(best_result.x[1])}, {np.round(best_result.x[2])}")
        print(
            f"New Outer Tie Rod: {np.round(bes_result.x[3])}, {np.round(best_result.x[4])}, {np.round(best_result.x[5])}")

    def rodrigues_rotate(self, vec, axis, theta):

        axis = np.asarray(axis, dtype=float)
        v = np.asarray(vec, dtype=float)
        # ensure axis unit
        axis = axis / np.linalg.norm(axis)
        # Rodrigues formula
        # v_rot = v*cosθ + (k × v)*sinθ + k*(k·v)*(1-cosθ)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        k = axis
        v_cos = v * cos_t
        v_cross = np.cross(k, v) * sin_t
        v_dot = k * (np.dot(k, v) * (1.0 - cos_t))
        return v_cos + v_cross + v_dot

    def solve_by_angles(self, motion_value, heave_mode='avg', prev_thetas=None):

        upper_inner = (
            self.upper_wishbone['chassis_fore'] - self.upper_wishbone['chassis_rear'])/2
        lower_inner = (
            self.lower_wishbone['chassis_fore'] - self.lower_wishbone['chassis_rear'])/2
        initial_upper_out = self.upper_wishbone['upright']
        initial_lower_out = self.lower_wishbone['upright']
        u_axis = self.upper_wishbone['chassis_fore'] - \
            self.upper_wishbone['chassis_rear']
        l_axis = self.lower_wishbone['chassis_fore'] - \
            self.lower_wishbone['chassis_rear']
        upright_length = np.linalg.norm(initial_upper_out - initial_lower_out)

        upper_inner = np.asarray(upper_inner, dtype=float)
        lower_inner = np.asarray(lower_inner, dtype=float)
        initial_upper_out = np.asarray(initial_upper_out, dtype=float)
        initial_lower_out = np.asarray(initial_lower_out, dtype=float)
        u_axis = np.asarray(u_axis, dtype=float)
        l_axis = np.asarray(l_axis, dtype=float)

        # normalize axes
        if np.linalg.norm(u_axis) < 1e-12 or np.linalg.norm(l_axis) < 1e-12:
            raise ValueError(
                "Hinge axis is zero-length; provide valid axis vectors.")
        u_axis = u_axis / np.linalg.norm(u_axis)
        l_axis = l_axis / np.linalg.norm(l_axis)

        # precompute vectors from inner pivots to outer points
        r_u0 = initial_upper_out - upper_inner
        r_l0 = initial_lower_out - lower_inner

        upright_length = np.linalg.norm(initial_upper_out - initial_lower_out)
        initial_avg_z = 0.5 * (initial_upper_out[2] + initial_lower_out[2])
        target_avg_z = initial_avg_z - motion_value

        # variables: theta_u, theta_l
        if isinstance(prev_thetas, (list, tuple, np.ndarray)) and len(prev_thetas) == 2:
            x0 = np.array(prev_thetas, dtype=float)
        else:
            x0 = np.array([0.0, 0.0], dtype=float)  # start near zero rotation

        def fun(thetas):
            tu, tl = float(thetas[0]), float(thetas[1])
            # rotate r_u0, r_l0 about respective axes
            r_u = self.rodrigues_rotate(r_u0, u_axis, tu)
            r_l = self.rodrigues_rotate(r_l0, l_axis, tl)
            upper = upper_inner + r_u
            lower = lower_inner + r_l

            # residual 1: upright length
            res1 = np.linalg.norm(upper - lower) - upright_length

            # residual 2: heave constraint
            if heave_mode == 'avg':
                res2 = 0.5 * (upper[2] + lower[2]) - target_avg_z
            elif heave_mode == 'lower':
                res2 = lower[2] - (initial_lower_out[2] + motion_value)
            else:
                raise ValueError("heave_mode must be 'avg' or 'lower'")

            return np.array([res1, res2], dtype=float)

        sol = least_squares(fun, x0, method='trf', ftol=1e-12,
                            xtol=1e-12, gtol=1e-12, max_nfev=5000)

        tu, tl = sol.x
        r_u = self.rodrigues_rotate(r_u0, u_axis, tu)
        r_l = self.rodrigues_rotate(r_l0, l_axis, tl)
        upper = upper_inner + r_u
        lower = lower_inner + r_l

        return upper, lower, np.array([tu, tl]), sol.success

    def solve_tie_rod(self, new_upper_out, new_lower_out, prev_tro=None):

        tri = self.tie_rod['inner']
        tro_init = self.tie_rod['outer']
        u_init = self.upper_wishbone['upright']
        l_init = self.lower_wishbone['upright']

        tie_rod_length = np.linalg.norm(tri - tro_init)
        dist_u_tro = np.linalg.norm(u_init - tro_init)
        dist_l_tro = np.linalg.norm(l_init - tro_init)

        def equations(vars):
            tro = np.array(vars, dtype=float)

            eq1 = np.linalg.norm(tro - tri) - tie_rod_length

            eq2 = np.linalg.norm(tro - new_upper_out) - dist_u_tro

            eq3 = np.linalg.norm(tro - new_lower_out) - dist_l_tro

            return np.array([eq1, eq2, eq3], dtype=float)

        x0 = prev_tro if prev_tro is not None else tro_init

        res = least_squares(equations, x0, method='trf',
                            ftol=1e-12, xtol=1e-12, gtol=1e-12)

        if not res.success:
            print("[Tie Rod Solver] Failed to converge. Check geometry limits.")

        return res.x, res.success

    def get_bump_steer(self, upper_out, lower_out, tro_out, static_toe_angle=None):
        kingpin_vec = upper_out - lower_out
        tierod_vec = tro_out - lower_out

        normal_vec = np.cross(kingpin_vec, tierod_vec)

        nx = normal_vec[0]
        ny = normal_vec[1]

        current_yaw_rad = np.arctan2(nx, ny)
        current_yaw_deg = 180-np.degrees(current_yaw_rad)

        if static_toe_angle is not None:
            delta_toe = current_yaw_deg - static_toe_angle
            bump_steer = (delta_toe + 180) % 360 - 180
            return bump_steer

        return current_yaw_deg

    def toe_optimizer(self, initial_guess=None, target_bump_steer=0.0):

        def objective(vars):
            self.tie_rod['inner'] = np.array([vars[0], vars[1], vars[2]])
            self.tie_rod['outer'] = np.array([vars[3], vars[4], vars[5]])

            total_error = 0.0

            u_stat, l_stat, _, stat_success = self.solve_by_angles(0.0)
            if not stat_success:
                return 1e6
            tro_stat, tro_success = self.solve_tie_rod(u_stat, l_stat)
            if not tro_success:
                return 1e6

            static_toe = self.get_bump_steer(u_stat, l_stat, tro_stat)

            heave_range = range(self.minimum_heave,
                                self.maximum_heave + 1, self.step)

            for heave in heave_range:
                if heave == 0:
                    continue

                upper, lower, _, w_success = self.solve_by_angles(heave)
                tie_rod_out, t_success = self.solve_tie_rod(upper, lower)
                if not w_success or not t_success:
                    return 1e6

                bump_steer = self.get_bump_steer(
                    upper, lower, tie_rod_out, static_toe_angle=static_toe)
                total_error += (bump_steer - target_bump_steer)**2

            return total_error

        if initial_guess is not None and len(initial_guess) == 6:
            x0 = np.array(initial_guess, dtype=float)
            print("Using custom initial guess...")
        else:
            x0 = np.concatenate((self.tie_rod['inner'], self.tie_rod['outer']))
            print("Using current tie rod coordinates as initial guess...")

        # Can move slightly fore/aft on the chassis
        inner_x_bounds = (-150.0, 0)
        inner_y_bounds = (195, 210)  # Rack width adjustment
        # LOCKED! Steering rack height is fixed at Z=150
        inner_z_bounds = (90, 300.0)

        # Outer Tie Rod (Upright Mount)
        outer_x_bounds = (-110.0, 0)  # Steering arm length
        outer_y_bounds = (500.0, 600.0)  # Wheel clearance limits
        outer_z_bounds = (90.0, 200.0)  # Can be shimmed up and down by 40mm

        # Package them into a tuple of tuples
        bnds = (inner_x_bounds, inner_y_bounds, inner_z_bounds,
                outer_x_bounds, outer_y_bounds, outer_z_bounds)

        print("Hunting for optimal Tie Rod coordinates within packaging limits...")

        # --- NEW: Switch method to SLSQP and pass the bounds ---
        res = minimize(objective, x0, method='SLSQP', bounds=bnds,
                       options={'maxiter': 10000, 'ftol': 1e-6})

        if res.success:

            self.tie_rod['inner'] = res.x[0:3]
            self.tie_rod['outer'] = res.x[3:6]

        else:
            print("Optimization failed to converge. Try a different initial guess.")

        return res


if __name__ == "__main__":
    optimizer = ToeOptimizer()
