import math
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize, fsolve, least_squares
from scipy.interpolate import make_interp_spline, BSpline, interp1d
import csv
from scipy.optimize import differential_evolution


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

            'inner': np.array([-10.0, 195.0, 250.0], dtype=float),
            'outer': np.array([-46.0, 596.0, 156.0], dtype=float)
        }
        self.hub = np.array([0, 613.227, 203.192])

        self.minimum_heave = -30
        self.maximum_heave = 30
        self.step = 1
        self.fixed_tr_length = np.linalg.norm(self.tie_rod['outer'] - self.tie_rod['inner'])
        self.fixed_dist_u_tro = np.linalg.norm(self.upper_wishbone['upright'] - self.tie_rod['outer'])
        self.fixed_dist_l_tro = np.linalg.norm(self.lower_wishbone['upright'] - self.tie_rod['outer'])
        print("--- Baseline Bump Steer Curve ---")

        u_stat, l_stat, _, _ = self.solve_by_angles(0.0)
        tro_stat, _ = self.solve_tie_rod(u_stat, l_stat)
        hub_stat, _ = self.solve_hub(u_stat, l_stat)
        
        static_R = self.get_upright_rotation_matrix(u_stat, l_stat, tro_stat, hub_stat)
        
        curve_results = {}
        prev_w_thetas = np.array([0.0, 0.0])
        prev_tro = tro_stat
        for heave in range(0, self.maximum_heave + 1, self.step):
            upper, lower, thetas, w_succ = self.solve_by_angles(heave, prev_thetas=prev_w_thetas)
            tro, t_succ = self.solve_tie_rod(upper, lower, prev_tro=prev_tro)
            hub, _ = self.solve_hub(upper, lower, prev_hub=self.hub)
            current_tr_length = np.linalg.norm(tro - self.tie_rod['inner'])
            if w_succ and t_succ:
                bump_steer = self.get_bump_steer(upper, lower, tro, hub, static_R)
                curve_results[heave] = bump_steer
                prev_w_thetas = thetas
                prev_tro = tro
            else:
                curve_results[heave] = None 

        prev_w_thetas = np.array([0.0, 0.0]) 
        prev_tro = tro_stat 
        for heave in range(-self.step, self.minimum_heave - 1, -self.step):
            upper, lower, thetas, w_succ = self.solve_by_angles(heave, prev_thetas=prev_w_thetas)
            tro, t_succ = self.solve_tie_rod(upper, lower, prev_tro=prev_tro)
            current_tr_length = np.linalg.norm(tro - self.tie_rod['inner'])
            if w_succ and t_succ:
                bump_steer = self.get_bump_steer(upper, lower, tro, hub, static_R)
                curve_results[heave] = bump_steer
                prev_w_thetas = thetas
                prev_tro = tro
            else:
                curve_results[heave] = None

        for heave in range(self.minimum_heave, self.maximum_heave + 1, self.step):
            val = curve_results.get(heave)
            if val is not None:
                print(f"Heave {heave:3}mm | Toe Change: {val:.4f} deg")
            else:
                print(f"Heave {heave:3}mm | KINEMATIC BIND (Solver Failed)")

        best_result = self.toe_optimizer(self.tie_rod['inner'].tolist() + self.tie_rod['outer'].tolist())
        print(best_result)
        for i in range(50):
            init = best_result.x + np.random.normal(scale=5.0, size=6)
            res = self.toe_optimizer(init)
            print(res)
            if res.fun < best_result.fun:
                best_result = res
                print(best_result)

        print(
            f"New Inner Tie Rod: {np.round(best_result.x[0])}, {np.round(best_result.x[1])}, {np.round(best_result.x[2])}")
        print(
            f"New Outer Tie Rod: {np.round(best_result.x[3])}, {np.round(best_result.x[4])}, {np.round(best_result.x[5])}")

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


        upper_inner = (self.upper_wishbone['chassis_fore'] + self.upper_wishbone['chassis_rear'])/2
        lower_inner = (self.lower_wishbone['chassis_fore'] + self.lower_wishbone['chassis_rear'])/2
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

        tie_rod_length = self.fixed_tr_length
        dist_u_tro = self.fixed_dist_u_tro
        dist_l_tro = self.fixed_dist_l_tro

        def equations(vars):
            tro = np.array(vars, dtype=float)

            eq1 = np.linalg.norm(tro - tri)**2 - tie_rod_length**2

            eq2 = np.linalg.norm(tro - new_upper_out)**2 - dist_u_tro**2

            eq3 = np.linalg.norm(tro - new_lower_out)**2 - dist_l_tro**2

            return np.array([eq1, eq2, eq3], dtype=float)

        x0 = prev_tro if prev_tro is not None else tro_init

        res = least_squares(equations, x0, method='trf',
                            ftol=1e-12, xtol=1e-12, gtol=1e-12)

        if not res.success:
            print("[Tie Rod Solver] Failed to converge. Check geometry limits.")

        return res.x, res.success

    def solve_hub(self, new_upper_out, new_lower_out, prev_hub=None):
        
        hub_init = self.hub
        u_init = self.upper_wishbone['upright']
        l_init = self.lower_wishbone['upright']

        dist_u_hub = np.linalg.norm(u_init - hub_init)
        dist_l_hub = np.linalg.norm(l_init - hub_init)

        def equations(vars):
            hub = np.array(vars, dtype=float)

            eq1 = np.linalg.norm(hub - new_upper_out)**2 - dist_u_hub**2

            eq2 = np.linalg.norm(hub - new_lower_out)**2 - dist_l_hub**2

            return np.array([eq1, eq2], dtype=float)

        x0 = prev_hub if prev_hub is not None else hub_init

        res = least_squares(equations, x0, method='trf',
                            ftol=1e-12, xtol=1e-12, gtol=1e-12)

        if not res.success:
            print("[Hub Solver] Failed to converge. Check geometry limits.")

        return res.x, res.success
    
    def get_bump_steer(self, upper_out, lower_out, tro_out, hub_out, static_R=None):
        current_R = self.get_upright_rotation_matrix(upper_out, lower_out, tro_out, hub_out)
        
        if static_R is None:
            return 0.0 
            
        R_delta = current_R @ static_R.T
        
        static_heading = np.array([1.0, 0.0, 0.0])
        current_heading = R_delta @ static_heading
        toe_deg = np.degrees(np.arctan2(current_heading[1], current_heading[0]))
        
        return -toe_deg

    def toe_optimizer(self, initial_guess=None, target_bump_steer=0.0):
        # 1. Initialize x0 safely
        if initial_guess is not None and len(initial_guess) == 6:
            x0 = np.array(initial_guess, dtype=float)
        else:
            x0 = np.concatenate((self.tie_rod['inner'], self.tie_rod['outer']))

        self.fixed_tr_length = np.linalg.norm(self.tie_rod['outer'] - self.tie_rod['inner'])
        self.fixed_dist_u_tro = np.linalg.norm(self.upper_wishbone['upright'] - self.tie_rod['outer'])
        self.fixed_dist_l_tro = np.linalg.norm(self.lower_wishbone['upright'] - self.tie_rod['outer'])
        

        # Bounds definition
        bnds = [(-150, 0), (190, 220), (90, 300),  # Inner X, Y, Z
                (-110, 0), (500, 600), (90, 200)] # Outer X, Y, Z

        # Using a larger 'eps' (step size) helps SLSQP see past internal solver noise
        res = differential_evolution(
        self.objective,
        bnds,
        strategy='best1bin',
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        workers=1  # Uses all CPU cores
    )

        if res.success:
            self.tie_rod['inner'] = res.x[0:3]
            self.tie_rod['outer'] = res.x[3:6]
        return res
    
    def objective(self, vars, target_bump_steer=0.0):
            # Update the model coordinates for this iteration
            self.tie_rod['inner'] = vars[0:3]
            self.tie_rod['outer'] = vars[3:6]

            total_error = 0.0
            
            # Solve static reference
            u_stat, l_stat, _, stat_succ = self.solve_by_angles(0.0)
            if not stat_succ: return 1e8
            tro_stat, t_succ = self.solve_tie_rod(u_stat, l_stat)
            hub_stat, h_succ = self.solve_hub(u_stat, l_stat)
            if not (t_succ and h_succ): return 1e8

            static_R = self.get_upright_rotation_matrix(u_stat, l_stat, tro_stat, hub_stat)
            
            # Combine bump and rebound into one clean list to check
            heave_range = list(range(self.minimum_heave, self.maximum_heave + 1, 2)) # Step by 2 for speed
            
            prev_w_thetas = np.array([0.0, 0.0])
            prev_tro_guess = tro_stat

            for heave in heave_range:
                if heave == 0: continue
                
                u, l, thetas, w_succ = self.solve_by_angles(heave, prev_thetas=prev_w_thetas)
                tro, t_succ = self.solve_tie_rod(u, l, prev_tro=prev_tro_guess)
                hub, h_succ = self.solve_hub(u, l, prev_hub=self.hub)

                if w_succ and t_succ and h_succ:
                    steer = self.get_bump_steer(u, l, tro, hub, static_R)
                    
                    # --- FIX FOR THE 180-DEGREE FLIP ---
                    # If the solver jumps ~180 degrees, normalize it back to near zero
                    if steer > 90: steer -= 180
                    elif steer < -90: steer += 180
                    
                    total_error += (steer - target_bump_steer)**2
                    
                    # Seed next iteration for stability
                    prev_w_thetas = thetas
                    prev_tro_guess = tro
                else:
                    print(f"[Objective] Solver failed at heave {heave}mm. Penalizing heavily.")
                    total_error += 1e8  # Penalty for kinematic binding

            return total_error
    def get_upright_rotation_matrix(self, upper, lower, tro, hub):

        z_axis = upper - lower
        z_axis /= np.linalg.norm(z_axis)
        
        spindle_vec = hub - lower
        spindle_vec /= np.linalg.norm(spindle_vec)
        
        x_axis = spindle_vec - np.dot(spindle_vec, z_axis) * z_axis
        x_axis /= np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        
        return np.column_stack((x_axis, y_axis, z_axis))



if __name__ == "__main__":
    optimizer = ToeOptimizer()
