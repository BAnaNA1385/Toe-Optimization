import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# ---------------------------------------
# Rotation matrix from XYZ Euler angles
# ---------------------------------------
def rotation_matrix(phi, theta, psi):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi),  np.cos(phi)]
    ])
    Ry = np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    Rz = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi),  np.cos(psi), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx


class RigidUprightModel:
    def __init__(self):

        # ----------------------------
        # CHASSIS HARDPOINTS
        # ----------------------------
        self.UF = np.array([200, 300, 400])
        self.UR = np.array([-200, 300, 400])
        self.LF = np.array([200, 300, 100])
        self.LR = np.array([-200, 300, 100])
        self.TR_in = np.array([0, 250, 200])

        # ----------------------------
        # UPRIGHT LOCAL COORDINATES
        # (relative to wheel center)
        # ----------------------------
        self.p_U = np.array([0, -50, 100])
        self.p_L = np.array([0, -50, -100])
        self.p_TR = np.array([50, -50, 0])
        self.p_WC = np.array([0, 0, 0])

        # Static pose
        self.r_static = np.array([0, 650, 250])
        self.R_static = np.eye(3)

        # Compute static global points
        self.compute_static_lengths()

    def compute_static_lengths(self):
        def global_point(p):
            return self.r_static + self.R_static @ p

        self.lengths = {
            "UF": np.linalg.norm(global_point(self.p_U) - self.UF),
            "UR": np.linalg.norm(global_point(self.p_U) - self.UR),
            "LF": np.linalg.norm(global_point(self.p_L) - self.LF),
            "LR": np.linalg.norm(global_point(self.p_L) - self.LR),
            "TR": np.linalg.norm(global_point(self.p_TR) - self.TR_in),
        }

    # ---------------------------------------
    # Residuals for rigid body constraints
    # ---------------------------------------
    def residuals(self, x, heave):
        rx, ry, rz, phi, theta, psi = x

        r = np.array([rx, ry, rz])
        R = rotation_matrix(phi, theta, psi)

        # Global upright points
        P_U = r + R @ self.p_U
        P_L = r + R @ self.p_L
        P_TR = r + R @ self.p_TR
        P_WC = r + R @ self.p_WC

        res = []

        # Wishbones
        res.append(np.linalg.norm(P_U - self.UF) - self.lengths["UF"])
        res.append(np.linalg.norm(P_U - self.UR) - self.lengths["UR"])
        res.append(np.linalg.norm(P_L - self.LF) - self.lengths["LF"])
        res.append(np.linalg.norm(P_L - self.LR) - self.lengths["LR"])

        # Tie rod
        res.append(np.linalg.norm(P_TR - self.TR_in) - self.lengths["TR"])

        # Heave constraint (wheel center vertical displacement)
        res.append(P_WC[2] - (self.r_static[2] + heave))

        return res

    # ---------------------------------------
    # Solve for pose at given heave
    # ---------------------------------------
    def solve_heave(self, heave):
        x0 = np.array([
            *self.r_static,
            0.0, 0.0, 0.0
        ])

        sol = least_squares(
            self.residuals,
            x0,
            args=(heave,),
            xtol=1e-12,
            ftol=1e-12
        )

        return sol.x

    # ---------------------------------------
    # Compute toe from rotation matrix
    # ---------------------------------------
    def compute_toe(self, solution):
        phi, theta, psi = solution[3:]
        R = rotation_matrix(phi, theta, psi)

        # Wheel forward direction (local X-axis)
        v = R @ np.array([1, 0, 0])

        toe = np.arctan2(v[1], v[0])
        return np.degrees(toe)


# ==========================================
# Run simulation
# ==========================================

model = RigidUprightModel()

heave_range = np.linspace(-30, 30, 50)
toe_values = []

for h in heave_range:
    sol = model.solve_heave(h)
    toe_values.append(model.compute_toe(sol))

plt.plot(heave_range, toe_values)
plt.xlabel("Heave (mm)")
plt.ylabel("Toe Angle (deg)")
plt.title("Toe vs Heave (Rigid Body Model)")
plt.grid()
plt.show()
