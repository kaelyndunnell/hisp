"""Create ITER scenario bins directly without CSV loading.

Replicates the bins from PFC-Tritium-Transport/simulations/scenarioA/input_table.csv
and materials.csv using real PFC-TT Bin, Material, BinConfiguration, Trap objects.
"""
from hisp.bin import Bin, BinConfiguration, Material

# Trigger PFC-TT path resolution (done by hisp.bin above), then import Trap
from materials.materials import Trap


# === MATERIALS (from scenarioA/materials.csv) ===

W_MATERIAL = Material(
    name="W",
    Mat_density=6.3382e28,
    D0=2.06e-7,
    E_D=0.28,
    K_R=7.94e-17,
    E_R=-2.0,
    N_traps=2,
    traps=[
        Trap(Trap_density=1e-4, k_0=3.58e-16, E_k=0.28, p_0=1e13, E_p=0.85),
        Trap(Trap_density=1e-4, k_0=3.58e-16, E_k=0.28, p_0=1e13, E_p=1.0),
    ],
)

B_MATERIAL = Material(
    name="B",
    Mat_density=1.34e29,
    D0=1.07e-6,
    E_D=0.3,
    K_R=0.0,
    E_R=0.0,
    N_traps=4,
    traps=[
        Trap(Trap_density=6.87e-1, k_0=7.4627e-17, E_k=0.3, p_0=1e13, E_p=1.052),
        Trap(Trap_density=5.21e-1, k_0=7.4627e-17, E_k=0.3, p_0=1e13, E_p=1.199),
        Trap(Trap_density=2.47e-1, k_0=7.4627e-17, E_k=0.3, p_0=1e13, E_p=1.389),
        Trap(Trap_density=1.28e-1, k_0=7.4627e-17, E_k=0.3, p_0=1e13, E_p=1.589),
    ],
)

_MATERIALS = {"W": W_MATERIAL, "B": B_MATERIAL}


# === BIN TABLE (from scenarioA/input_table.csv) ===
# Each tuple: (flux_id, z_start, r_start, z_end, r_end, material_name,
#              thickness, cu_thickness, mode, parent_area, surface_area,
#              f_fraction, location, rtol, atol, fp_max_stepsize,
#              max_stepsize_no_fp, bc_pfs, bc_rear)

_BIN_DATA = [
    (0, -2.51, 4.1, -1.5, 4.1, "W", 0.006, 0.002, "high_wetted", 26, 0.886, 0.919, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (0, -2.51, 4.1, -1.5, 4.1, "W", 0.006, 0.002, "low_wetted", 26, 0.0999, 0.0808, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (0, -2.51, 4.1, -1.5, 4.1, "W", 0.006, 0.002, "shadowed", 26, 25.1, 0, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (1, -1.5, 4.1, -0.487, 4.1, "W", 0.006, 0.002, "high_wetted", 26.1, 3.3, 0.978, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (1, -1.5, 4.1, -0.487, 4.1, "W", 0.006, 0.002, "low_wetted", 26.1, 0.137, 0.0223, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (1, -1.5, 4.1, -0.487, 4.1, "W", 0.006, 0.002, "shadowed", 26.1, 22.7, 0, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (2, -0.487, 4.1, 0.528, 4.1, "W", 0.01, 0.004, "high_wetted", 26.1, 1.75, 0.435, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (2, -0.487, 4.1, 0.528, 4.1, "W", 0.01, 0.004, "low_wetted", 26.1, 4.46, 0.565, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (2, -0.487, 4.1, 0.528, 4.1, "W", 0.01, 0.004, "shadowed", 26.1, 19.9, 0, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (3, 0.528, 4.1, 1.54, 4.1, "W", 0.01, 0.004, "high_wetted", 26.2, 0.475, 0.284, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (3, 0.528, 4.1, 1.54, 4.1, "W", 0.01, 0.004, "low_wetted", 26.2, 3.18, 0.716, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (3, 0.528, 4.1, 1.54, 4.1, "W", 0.01, 0.004, "shadowed", 26.2, 22.5, 0, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (4, 1.54, 4.1, 2.56, 4.1, "W", 0.01, 0.004, "high_wetted", 26.1, 3.9, 0.988, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (4, 1.54, 4.1, 2.56, 4.1, "W", 0.01, 0.004, "low_wetted", 26.1, 0.164, 0.0115, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (4, 1.54, 4.1, 2.56, 4.1, "W", 0.01, 0.004, "shadowed", 26.1, 22, 0, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (5, 2.56, 4.1, 3.57, 4.12, "W", 0.012, 0.004, "high_wetted", 26.1, 0.31, 0.981, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (5, 2.56, 4.1, 3.57, 4.12, "W", 0.012, 0.004, "low_wetted", 26.1, 0.00947, 0.0189, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (5, 2.56, 4.1, 3.57, 4.12, "W", 0.012, 0.004, "shadowed", 26.1, 25.8, 0, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (6, 3.57, 4.12, 4.33, 4.33, "W", 0.012, 0.005, "high_wetted", 20.8, 0.31, 0.981, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (6, 3.57, 4.12, 4.33, 4.33, "W", 0.012, 0.005, "low_wetted", 20.8, 0.00947, 0.0189, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (6, 3.57, 4.12, 4.33, 4.33, "W", 0.012, 0.005, "shadowed", 20.8, 20.5, 0, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (7, 4.33, 4.33, 4.7, 4.93, "W", 0.012, 0.005, "high_wetted", 20.7, 1.94, 0.992, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (7, 4.33, 4.33, 4.7, 4.93, "W", 0.012, 0.005, "low_wetted", 20.7, 0.0576, 0.00813, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (7, 4.33, 4.33, 4.7, 4.93, "W", 0.012, 0.005, "shadowed", 20.7, 18.7, 0, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (8, 4.7, 4.93, 4.52, 5.75, "W", 0.012, 0.005, "high_wetted", 28.2, 4.86, 0.973, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (8, 4.7, 4.93, 4.52, 5.75, "W", 0.012, 0.005, "low_wetted", 28.2, 0.311, 0.0265, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (8, 4.7, 4.93, 4.52, 5.75, "W", 0.012, 0.005, "shadowed", 28.2, 23, 0, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (9, 4.52, 5.75, 3.94, 6.51, "W", 0.012, 0.002, "high_wetted", 24.5, 4.86, 0.973, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (9, 4.52, 5.75, 3.94, 6.51, "W", 0.012, 0.002, "low_wetted", 24.5, 0.311, 0.0265, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (9, 4.52, 5.75, 3.94, 6.51, "W", 0.012, 0.002, "shadowed", 24.5, 19.3, 0, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (10, 3.94, 6.51, 3.16, 7.4, "W", 0.012, 0.002, "high_wetted", 51.6, 4.86, 0.973, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (10, 3.94, 6.51, 3.16, 7.4, "B", 1e-7, 0, "low_wetted", 51.6, 0.311, 0.0265, "FW", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (10, 3.94, 6.51, 3.16, 7.4, "B", 1e-6, 0, "shadowed", 51.6, 46.4, 0, "FW", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (11, 3.16, 7.4, 2.44, 7.9, "W", 0.012, 0.002, "high_wetted", 42, 0.00359, 0.00522, "FW", 1e-13, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (11, 3.16, 7.4, 2.44, 7.9, "B", 1e-7, 0, "low_wetted", 42, 5.2, 0.995, "FW", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (11, 3.16, 7.4, 2.44, 7.9, "B", 1e-6, 0, "shadowed", 42, 36.8, 0, "FW", 1e-12, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (12, 2.44, 7.9, 1.66, 8.26, "W", 0.012, 0.004, "high_wetted", 43.6, 0.14, 0.741, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (12, 2.44, 7.9, 1.66, 8.26, "B", 1e-7, 0, "low_wetted", 43.6, 0.127, 0.259, "FW", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (12, 2.44, 7.9, 1.66, 8.26, "B", 1e-6, 0, "shadowed", 43.6, 43.3, 0, "FW", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (13, 1.66, 8.26, 0.613, 8.38, "W", 0.006, 0.002, "wetted", 27.6, 4.44, 1, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (13, 1.66, 8.26, 0.613, 8.38, "B", 1e-6, 0, "shadowed", 27.6, 23.2, 0, "FW", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (14, 0.613, 8.38, -0.441, 8.29, "W", 0.006, 0.002, "wetted", 27.7, 4.55, 1, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (14, 0.613, 8.38, -0.441, 8.29, "B", 1e-6, 0, "shadowed", 27.7, 23.2, 0, "FW", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (15, -0.441, 8.29, -1.36, 7.89, "W", 0.006, 0.005, "high_wetted", 50.7, 0.0899, 0.0514, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (15, -0.441, 8.29, -1.36, 7.89, "B", 1e-7, 0, "low_wetted", 50.7, 5.42, 0.949, "FW", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (15, -0.441, 8.29, -1.36, 7.89, "B", 1e-6, 0, "shadowed", 50.7, 45.2, 0, "FW", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (16, -1.36, 7.89, -2.26, 7.27, "W", 0.006, 0.005, "wetted", 52.6, 3.52, 1, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (16, -1.36, 7.89, -2.26, 7.27, "B", 1e-6, 0, "shadowed", 52.6, 49.1, 0, "FW", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (17, -2.26, 7.27, -3.25, 6.06, "W", 0.012, 0.002, "wetted", 54.4, 8.34, 1, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (17, -2.26, 7.27, -3.25, 6.06, "W", 0.012, 0.002, "shadowed", 54.4, 46.1, 0, "FW", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (18, -3.25, 6.06, -3.35, 5.83, "B", 1e-6, 0, "", 9.37, 9.37, 1, "Div", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (19, -3.35, 5.83, -3.54, 5.66, "B", 1e-6, 0, "", 9.2, 9.2, 1, "Div", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (20, -3.54, 5.66, -3.77, 5.57, "B", 1e-6, 0, "", 8.71, 8.71, 1, "Div", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (21, -3.77, 5.57, -3.93, 5.56, "B", 1e-6, 0, "", 5.61, 5.61, 1, "Div", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (22, -3.93, 5.56, -4.01, 5.56, "W", 0.006, 0, "", 2.79, 2.79, 1, "Div", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (23, -4.01, 5.56, -4.08, 5.56, "W", 0.006, 0, "", 2.45, 2.45, 1, "Div", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (24, -4.08, 5.56, -4.14, 5.56, "W", 0.006, 0, "", 2.1, 2.1, 1, "Div", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (25, -4.14, 5.56, -4.2, 5.56, "W", 0.006, 0, "", 2.1, 2.1, 1, "Div", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (26, -4.2, 5.56, -4.26, 5.56, "W", 0.006, 0, "", 2.1, 2.1, 1, "Div", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (27, -4.26, 5.56, -4.32, 5.56, "W", 0.006, 0, "", 2.1, 2.1, 1, "Div", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (28, -4.32, 5.56, -4.38, 5.56, "W", 0.006, 0, "", 2.1, 2.1, 1, "Div", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (29, -4.38, 5.56, -4.44, 5.56, "W", 0.006, 0, "", 2.1, 2.1, 1, "Div", 1e-12, 1e11, 20, 2000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (30, -4.44, 5.56, -4.5, 5.56, "W", 0.006, 0, "", 2.1, 2.1, 1, "Div", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (31, -4.5, 5.56, -4.58, 5.56, "W", 0.006, 0, "", 2.79, 2.79, 1, "Div", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (32, -4.58, 5.56, -4.41, 5.41, "B", 1e-6, 0, "", 7.81, 7.81, 1, "Div", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (33, -4.41, 5.41, -4.27, 5.27, "B", 1e-6, 0, "", 6.64, 6.64, 1, "Div", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (34, -3.99, 5.25, -3.86, 5.16, "B", 1e-6, 0, "", 5.17, 5.17, 1, "Div", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (35, -3.86, 5.16, -3.77, 5.04, "B", 1e-6, 0, "", 4.81, 4.81, 1, "Div", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (36, -3.77, 5.04, -3.73, 4.94, "B", 1e-6, 0, "", 3.38, 3.38, 1, "Div", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (37, -3.73, 4.94, -3.71, 4.82, "B", 1e-6, 0, "", 3.73, 3.73, 1, "Div", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (38, -3.71, 4.82, -3.72, 4.74, "B", 1e-6, 0, "", 2.42, 2.42, 1, "Div", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (39, -3.72, 4.74, -3.75, 4.64, "B", 1e-6, 0, "", 3.08, 3.08, 1, "Div", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (40, -3.91, 4.49, -3.9, 4.39, "B", 1e-6, 0, "", 2.8, 2.8, 1, "Div", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (41, -3.9, 4.39, -3.9, 4.29, "B", 5e-6, 0, "", 2.73, 2.73, 1, "Div", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (42, -3.9, 4.29, -3.89, 4.22, "B", 5e-6, 0, "", 1.89, 1.89, 1, "Div", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (43, -3.89, 4.22, -3.92, 4.16, "B", 5e-6, 0, "", 1.77, 1.77, 1, "Div", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (44, -3.92, 4.16, -3.87, 4.18, "B", 5e-6, 0, "", 1.41, 1.41, 1, "Div", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (45, -3.87, 4.18, -3.81, 4.21, "W", 0.006, 0, "", 1.77, 1.77, 1, "Div", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (46, -3.81, 4.21, -3.76, 4.24, "W", 0.006, 0, "", 1.55, 1.55, 1, "Div", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (47, -3.76, 4.24, -3.71, 4.26, "W", 0.006, 0, "", 1.44, 1.44, 1, "Div", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (48, -3.71, 4.26, -3.64, 4.29, "W", 0.006, 0, "", 2.05, 2.05, 1, "Div", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (49, -3.64, 4.29, -3.59, 4.32, "W", 0.006, 0, "", 1.58, 1.58, 1, "Div", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (50, -3.59, 4.32, -3.54, 4.35, "W", 0.006, 0, "", 1.59, 1.59, 1, "Div", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (51, -3.54, 4.35, -3.48, 4.37, "W", 0.006, 0, "", 1.73, 1.73, 1, "Div", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (52, -3.48, 4.37, -3.43, 4.4, "W", 0.006, 0, "", 1.61, 1.61, 1, "Div", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (53, -3.43, 4.4, -3.38, 4.42, "W", 0.006, 0, "", 1.49, 1.49, 1, "Div", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (54, -3.38, 4.42, -3.31, 4.46, "W", 0.006, 0, "", 2.25, 2.25, 1, "Div", 1e-12, 1e12, 20, 1000, "Robin - Surf. Rec. + Implantation", "Neumann - no flux"),
    (55, -3.31, 4.46, -3.24, 4.49, "B", 5e-6, 0, "", 2.14, 2.14, 1, "Div", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (56, -3.24, 4.49, -3.14, 4.51, "B", 5e-6, 0, "", 2.88, 2.88, 1, "Div", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (57, -3.14, 4.51, -3.04, 4.52, "B", 5e-6, 0, "", 2.85, 2.85, 1, "Div", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (58, -3.04, 4.52, -2.93, 4.5, "B", 5e-6, 0, "", 3.17, 3.17, 1, "Div", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (59, -2.93, 4.5, -2.79, 4.44, "B", 5e-6, 0, "", 4.28, 4.28, 1, "Div", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (60, -2.79, 4.44, -2.67, 4.35, "B", 5e-6, 0, "", 4.14, 4.14, 1, "Div", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
    (61, -2.67, 4.35, -2.51, 4.1, "B", 5e-6, 0, "", 7.88, 7.88, 1, "Div", 1e-13, 1e11, 20, 200, "Dirichlet - 0 concentration + Implantation", "Neumann - no flux"),
]


def make_iter_bins():
    """Create all ITER scenario A bins as a list of Bin objects.

    Returns:
        list[Bin]: 82 Bin objects replicating scenarioA/input_table.csv.
    """
    bins = []
    for row in _BIN_DATA:
        (flux_id, z_start, r_start, z_end, r_end, mat_name,
         thickness, cu_thickness, mode, parent_area, surface_area,
         f_fraction, location, rtol, atol, fp_max_stepsize,
         max_stepsize_no_fp, bc_pfs, bc_rear) = row

        config = BinConfiguration(
            rtol=rtol,
            atol=atol,
            fp_max_stepsize=fp_max_stepsize,
            max_stepsize_no_fp=max_stepsize_no_fp,
            bc_plasma_facing_surface=bc_pfs,
            bc_rear_surface=bc_rear,
        )

        bins.append(Bin(
            flux_id=flux_id,
            material=_MATERIALS[mat_name],
            thickness=thickness,
            cu_thickness=cu_thickness,
            mode=mode,
            parent_bin_surf_area=parent_area,
            surface_area=surface_area,
            f_ion_flux_fraction=f_fraction,
            location=location,
            z_start=z_start,
            r_start=r_start,
            z_end=z_end,
            r_end=r_end,
            bin_configuration=config,
            calculate_implantation_params=False,
        ))

    return bins
