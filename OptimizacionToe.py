import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, differential_evolution

class ToeOptimizer:
    def __init__(self):
        self.upper_wishbone = {
            'upright': np.array([0.0, 537.77, 290.0]),
            'chassis_fore': np.array([150.0, 195.0, 256.83]),
            'chassis_rear': np.array([-150.0, 195.0, 267.77])
        }
        self.lower_wishbone = {
            'upright': np.array([0.0, 585.0, 112.65]),
            'chassis_fore': np.array([150.0, 195.0, 116.52]),
            'chassis_rear': np.array([-150.0, 195.0, 100.44])
        }
        self.hub_init = np.array([0.0, 613.227, 203.192])

        self.active_inner = np.array([-10.0, 195.0, 250.0])
        self.active_tr_len = 0.0
        self.active_u_tro = 0.0
        self.active_l_tro = 0.0
        self.active_u_hub = np.linalg.norm(self.upper_wishbone['upright'] - self.hub_init)
        self.active_l_hub = np.linalg.norm(self.lower_wishbone['upright'] - self.hub_init)

    def rodrigues_rotate(self, vec, axis, theta):
        axis = axis / np.linalg.norm(axis)
        return vec * np.cos(theta) + np.cross(axis, vec) * np.sin(theta) + axis * (np.dot(axis, vec) * (1.0 - np.cos(theta)))

    def solve_by_angles(self, motion_value):
        u_in = (self.upper_wishbone['chassis_fore'] + self.upper_wishbone['chassis_rear']) / 2
        l_in = (self.lower_wishbone['chassis_fore'] + self.lower_wishbone['chassis_rear']) / 2
        u_ax = (self.upper_wishbone['chassis_fore'] - self.upper_wishbone['chassis_rear'])
        l_ax = (self.lower_wishbone['chassis_fore'] - self.lower_wishbone['chassis_rear'])
        
        r_u0 = self.upper_wishbone['upright'] - u_in
        r_l0 = self.lower_wishbone['upright'] - l_in
        up_len = np.linalg.norm(self.upper_wishbone['upright'] - self.lower_wishbone['upright'])
        target_z = 0.5 * (self.upper_wishbone['upright'][2] + self.lower_wishbone['upright'][2]) - motion_value

        def eq(t):
            u = u_in + self.rodrigues_rotate(r_u0, u_ax, t[0])
            l = l_in + self.rodrigues_rotate(r_l0, l_ax, t[1])
            return [np.linalg.norm(u - l) - up_len, 0.5 * (u[2] + l[2]) - target_z]

        res = least_squares(eq, [0.0, 0.0], ftol=1e-12)
        u_final = u_in + self.rodrigues_rotate(r_u0, u_ax, res.x[0])
        l_final = l_in + self.rodrigues_rotate(r_l0, l_ax, res.x[1])
        return u_final, l_final, res.success

    def solve_tie_rod(self, u, l, outer_guess=None):
        def eq(tro):
            return [np.linalg.norm(tro - self.active_inner) - self.active_tr_len,
                    np.linalg.norm(tro - u) - self.active_u_tro,
                    np.linalg.norm(tro - l) - self.active_l_tro]

        guess = outer_guess if outer_guess is not None else (u + l) / 2 + np.array([0, 20, 0])
        res = least_squares(eq, guess, ftol=1e-12)
        return res.x, res.success


    def solve_hub(self, u, l, tro):
        def eq(h):
            return [
                np.linalg.norm(h - u) - self.active_u_hub,
                np.linalg.norm(h - l) - self.active_l_hub
            ]
        res = least_squares(eq, tro, ftol=1e-12)
        return res.x, res.success
    
    def get_orientation(self, u, l, h):
        x_axis = (u - l) / np.linalg.norm(u - l)
        z_axis = np.cross(x_axis, h - u)
        z_axis /= np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, x_axis)
        return np.column_stack((x_axis, y_axis, z_axis))   
    
    def objective(self, vars):

        # Update active tie rod positions
        self.active_inner = vars[0:3]
        outer_guess = np.array(vars[3:6])
        
        self.active_tr_len = np.linalg.norm(outer_guess - self.active_inner)
        self.active_u_tro = np.linalg.norm(self.upper_wishbone['upright'] - outer_guess)
        self.active_l_tro = np.linalg.norm(self.lower_wishbone['upright'] - outer_guess)
        
        u_s, l_s, ok_s = self.solve_by_angles(0.0)
        if not ok_s:
            return 1e9

        tr_s, ok_tr = self.solve_tie_rod(u_s, l_s, outer_guess=outer_guess)
        if not ok_tr:
            return 1e9

        h_s, ok_h = self.solve_hub(u_s, l_s, tr_s)
        if not ok_h:
            return 1e9

        static_R = self.get_orientation(u_s, l_s, h_s)

        heaves = np.arange(-30, 31, 2) 
        steer_results = []

        prev_theta_guess = None
        prev_tr_guess = tr_s

        for h in heaves:
            u, l, ok_w = self.solve_by_angles(h)
            if not ok_w:
                return 1e9

            tr, ok_tr = self.solve_tie_rod(u, l, outer_guess=prev_tr_guess)
            if not ok_tr:
                return 1e9
            prev_tr_guess = tr

            hub, ok_h = self.solve_hub(u, l, tr)
            if not ok_h:
                return 1e9

            curr_R = self.get_orientation(u, l, hub)
            heading = (curr_R @ static_R.T) @ np.array([1.0, 0.0, 0.0])
            toe_deg = np.degrees(np.arctan2(heading[1], heading[0]))
            steer_results.append(toe_deg)

        steer_results = np.array(steer_results)


        rms_error = np.sqrt(np.mean(steer_results**2))         # RMS magnitude
        peak_error = np.max(steer_results) - np.min(steer_results)  # Peak-to-peak
        curvature = np.sum(np.abs(np.diff(steer_results, n=2)))    # Second derivative

        total_error = (rms_error * 100) + (peak_error * 50) + (curvature * 10)
        
        return total_error



    def optimize(self):

        bnds = [
            (-100, 50),  (195, 220), (100, 300),
            (-100, 50),  (550, 620), (100, 300)  
        ]
        
        print("Starting Deep Kinematic Optimization...")
        res = differential_evolution(
            self.objective, bnds, 
            strategy='best1bin',
            popsize=50,
            maxiter=100,
            tol=1e-6, 
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True,
            disp=True
        )
        
        if res.success:
            print(f"\nOptimization Successful! Error: {res.fun:.6f}")
            print(f"New Inner: {res.x[0:3]}")
            print(f"New Outer: {res.x[3:6]}")
        return res

if __name__ == "__main__":
    solver = ToeOptimizer()
    res =solver.optimize()
    print(res)