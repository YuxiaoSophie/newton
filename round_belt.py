from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.examples

SCRIPT_DIR = Path(__file__).resolve().parent
TASK_BOARD_URDF_DIR = SCRIPT_DIR / "task_board_urdf"

ROUND_BELT_XACRO = SCRIPT_DIR / "round_belt.urdf.xacro"
if not ROUND_BELT_XACRO.exists():
    ROUND_BELT_XACRO = TASK_BOARD_URDF_DIR / "round_belt.urdf.xacro"

TABLE_OBJ = TASK_BOARD_URDF_DIR / "common" / "table" / "table.obj"
BOARD_MESH = TASK_BOARD_URDF_DIR / "common" / "task_board_just_board.glb"

SMALL_DIR = TASK_BOARD_URDF_DIR / "round_belt_task" / "round_belt_task_board" / "small_round_pulley"
LARGE_DIR = TASK_BOARD_URDF_DIR / "round_belt_task" / "round_belt_task_board" / "large_round_pulley"
SMALL_BRACKET_MESH = SMALL_DIR / "slide_tensioner_bracket.gltf"
SMALL_BEARING_MESH = SMALL_DIR / "slide_tensioner_bearing.gltf"
SMALL_BOLT_MESH = SMALL_DIR / "slide_tensioner_bolt.gltf"
SMALL_HALF_MESH = SMALL_DIR / "small_round_pulley_half.obj"
LARGE_HALF_MESH = LARGE_DIR / "large_round_pulley_half.obj"

TABLE_LENGTH_X = 1.20
TABLE_WIDTH_Y = 0.70
TABLE_TOP_Z = 0.72
TABLE_THICKNESS = 0.04

BOARD_SIZE = 0.384
BOARD_THICKNESS = 0.010
BOARD_ROOT_X = 0.150
BOARD_ROOT_Y = -0.192
BOARD_ROOT_Z = TABLE_TOP_Z + BOARD_THICKNESS

# Belt dimension
BELT_OUTER_MAJOR_DIAMETER = 0.248
BELT_OUTER_MINOR_DIAMETER = 0.168
BELT_TUBE_DIAMETER = 0.0066
BELT_RADIUS = BELT_TUBE_DIAMETER * 0.5 
BELT_CENTER_X = -0.320
BELT_CENTER_Y = 0.000
BELT_CENTER_Z = TABLE_TOP_Z + BELT_RADIUS 
BELT_NUM_ELEMENTS = 48

## 22 grams

# Helpers
def quat_from_rpy(roll: float, pitch: float, yaw: float) -> wp.quat:
    """URDF-style roll-pitch-yaw quaternion."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return wp.quat(qx, qy, qz, qw)


def tf(xyz, rpy=(0.0, 0.0, 0.0)) -> wp.transform:
    return wp.transform(
        wp.vec3(float(xyz[0]), float(xyz[1]), float(xyz[2])),
        quat_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2])),
    )


def board_world(local_xyz):
    return (
        BOARD_ROOT_X + float(local_xyz[0]),
        BOARD_ROOT_Y + float(local_xyz[1]),
        BOARD_ROOT_Z + float(local_xyz[2]),
    )


def make_visual_cfg() -> newton.ModelBuilder.ShapeConfig:
    return newton.ModelBuilder.ShapeConfig(
        density=0.0,
        has_shape_collision=False,
        has_particle_collision=False,
        collision_group=0,
        is_visible=True,
    )


def make_robust_table_collision_cfg(visible=True) -> newton.ModelBuilder.ShapeConfig:
    return newton.ModelBuilder.ShapeConfig(
        density=0.0,
        ke=1.0e4,
        kd=5.0e2,
        mu=1.5,
        has_shape_collision=True,
        has_particle_collision=True,
        collision_group=1,
        is_visible=visible,
    )


def add_visual_mesh(
    builder: newton.ModelBuilder,
    path: Path,
    xyz,
    rpy=(0.0, 0.0, 0.0),
    scale=(1.0, 1.0, 1.0),
    color=(0.8, 0.8, 0.8),
    label="visual_mesh",
) -> bool:
    if not path.exists():
        print(f"[missing mesh] {path}")
        return False
    try:
        mesh = newton.Mesh.create_from_file(str(path), compute_inertia=False, is_solid=False)
        builder.add_shape_mesh(
            body=-1,
            xform=tf(xyz, rpy),
            mesh=mesh,
            scale=wp.vec3(float(scale[0]), float(scale[1]), float(scale[2])),
            cfg=make_visual_cfg(),
            color=wp.vec3(float(color[0]), float(color[1]), float(color[2])),
            label=label,
        )
        return True
    except Exception as e:
        print(f"[failed mesh] {path}: {e}")
        return False


def read_obj_as_triangle_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    indices = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                face = []
                for token in parts[1:]:
                    idx_str = token.split("/")[0]
                    idx = int(idx_str)
                    if idx < 0:
                        idx = len(vertices) + idx
                    else:
                        idx = idx - 1
                    face.append(idx)
                for i in range(1, len(face) - 1):
                    indices.extend([face[0], face[i], face[i + 1]])
    if len(vertices) == 0 or len(indices) == 0:
        raise RuntimeError(f"No usable vertices/faces in {path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(indices, dtype=np.int32)


def fit_table_vertices(vertices: np.ndarray) -> np.ndarray:
    v = vertices.astype(np.float32).copy()
    v_min = v.min(axis=0)
    v_max = v.max(axis=0)
    dims = np.maximum(v_max - v_min, 1.0e-8)
    center = 0.5 * (v_min + v_max)
    scale_x = TABLE_LENGTH_X / dims[0]
    scale_y = TABLE_WIDTH_Y / dims[1]
    scale = min(scale_x, scale_y)
    v = (v - center) * scale
    v[:, 2] -= v[:, 2].max()
    v[:, 2] += TABLE_TOP_Z
    return v


def add_table(builder: newton.ModelBuilder) -> None:
    table_mesh_loaded = False
    if TABLE_OBJ.exists():
        try:
            vertices, indices = read_obj_as_triangle_mesh(TABLE_OBJ)
            vertices = fit_table_vertices(vertices)
            mesh = newton.Mesh(vertices, indices, compute_inertia=False, is_solid=False)
            builder.add_shape_mesh(
                body=-1,
                xform=tf((0.0, 0.0, 0.0)),
                mesh=mesh,
                scale=wp.vec3(1.0, 1.0, 1.0),
                cfg=make_visual_cfg(),
                color=wp.vec3(0.55, 0.35, 0.14),
                label="common_table_obj_visual",
            )
            table_mesh_loaded = True
        except Exception as e:
            print(f"[TABLE] failed to load {TABLE_OBJ}: {e}")

    builder.add_shape_box(
        body=-1,
        xform=tf((0.0, 0.0, TABLE_TOP_Z - 0.5 * TABLE_THICKNESS)),
        hx=0.5 * TABLE_LENGTH_X,
        hy=0.5 * TABLE_WIDTH_Y,
        hz=0.5 * TABLE_THICKNESS,
        cfg=make_robust_table_collision_cfg(visible=not table_mesh_loaded),
        color=wp.vec3(0.55, 0.35, 0.14),
        label="tabletop_collision",
    )

    if not table_mesh_loaded:
        leg_cfg = make_visual_cfg()
        leg_hx, leg_hy = 0.025, 0.025
        leg_hz = 0.5 * (TABLE_TOP_Z - TABLE_THICKNESS)
        x_edge = 0.5 * TABLE_LENGTH_X - 0.08
        y_edge = 0.5 * TABLE_WIDTH_Y - 0.08
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                builder.add_shape_box(
                    body=-1,
                    xform=tf((sx * x_edge, sy * y_edge, leg_hz)),
                    hx=leg_hx, hy=leg_hy, hz=leg_hz,
                    cfg=leg_cfg, color=wp.vec3(0.35, 0.20, 0.08),
                    label="fallback_table_leg",
                )


def add_board(builder: newton.ModelBuilder) -> None:
    board_mesh_ok = add_visual_mesh(
        builder, BOARD_MESH, xyz=board_world((0.0, 0.0, 0.0)),
        rpy=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0),
        color=(0.80, 0.80, 0.80), label="task_board_mesh_visual",
    )
    builder.add_shape_box(
        body=-1,
        xform=tf(board_world((0.192, 0.192, -0.005))),
        hx=0.5 * BOARD_SIZE, hy=0.5 * BOARD_SIZE, hz=0.5 * BOARD_THICKNESS,
        cfg=make_robust_table_collision_cfg(visible=not board_mesh_ok),
        color=wp.vec3(0.82, 0.82, 0.82),
        label="board_collision_exact_xacro",
    )


def add_pulleys_from_xacro_poses(builder: newton.ModelBuilder) -> None:
    visual_cfg = make_visual_cfg()
    small_bracket_xyz = board_world((0.3504, 0.1964, 0.0))
    bracket_ok = add_visual_mesh(
        builder, SMALL_BRACKET_MESH, xyz=small_bracket_xyz,
        rpy=(math.pi / 2.0, 0.0, math.pi / 2.0), scale=(1.0, 1.0, 1.0),
        color=(0.10, 0.10, 0.10), label="small_round_pulley_bracket_visual_exact_xacro",
    )
    if not bracket_ok:
        builder.add_shape_box(
            body=-1, xform=tf(board_world((0.3504, 0.1964, 0.010))),
            hx=0.018, hy=0.020, hz=0.006, cfg=visual_cfg,
            color=wp.vec3(0.10, 0.10, 0.10), label="small_bracket_fallback_visual",
        )

    small_center_local = (0.3504 - 0.01200845, 0.1964 - 0.0004, 0.0248)
    small_center_xyz = board_world(small_center_local)
    add_visual_mesh(
        builder, SMALL_BEARING_MESH,
        xyz=board_world((small_center_local[0], small_center_local[1], small_center_local[2] + 0.0035)),
        rpy=(0.0, math.pi / 2.0, 0.0), scale=(1.0, 1.0, 1.0),
        color=(0.10, 0.10, 0.10), label="small_bearing_visual_exact_xacro",
    )
    add_visual_mesh(
        builder, SMALL_BOLT_MESH, xyz=small_center_xyz,
        rpy=(0.0, math.pi / 2.0, 0.0), scale=(1.0, 1.0, 1.0),
        color=(0.10, 0.10, 0.10), label="small_bolt_visual_exact_xacro",
    )
    small_half_1_ok = add_visual_mesh(
        builder, SMALL_HALF_MESH, xyz=small_center_xyz, rpy=(0.0, 0.0, 0.0),
        scale=(0.001, 0.001, 0.001), color=(0.80, 0.80, 0.80), label="small_pulley_first_half_exact_xacro",
    )
    small_half_2_ok = add_visual_mesh(
        builder, SMALL_HALF_MESH, xyz=small_center_xyz, rpy=(math.pi, 0.0, 0.0),
        scale=(0.001, 0.001, 0.001), color=(0.80, 0.80, 0.80), label="small_pulley_second_half_exact_xacro",
    )
    if not (small_half_1_ok and small_half_2_ok):
        builder.add_shape_cylinder(
            body=-1, xform=tf(small_center_xyz), radius=0.015, half_height=0.0045,
            cfg=visual_cfg, color=wp.vec3(0.80, 0.80, 0.80), label="small_pulley_fallback_cylinder_visual",
        )

    large_center_local = (0.140, 0.196, 0.0248)
    large_center_xyz = board_world(large_center_local)
    large_half_1_ok = add_visual_mesh(
        builder, LARGE_HALF_MESH, xyz=large_center_xyz, rpy=(0.0, 0.0, 0.0),
        scale=(0.001, 0.001, 0.001), color=(0.10, 0.10, 0.10), label="large_pulley_first_half_exact_xacro",
    )
    large_half_2_ok = add_visual_mesh(
        builder, LARGE_HALF_MESH, xyz=large_center_xyz, rpy=(math.pi, 0.0, 0.0),
        scale=(0.001, 0.001, 0.001), color=(0.10, 0.10, 0.10), label="large_pulley_second_half_exact_xacro",
    )
    if not (large_half_1_ok and large_half_2_ok):
        builder.add_shape_cylinder(
            body=-1, xform=tf(large_center_xyz), radius=0.035, half_height=0.006,
            cfg=visual_cfg, color=wp.vec3(0.05, 0.05, 0.05), label="large_pulley_fallback_cylinder_visual",
        )


def create_ellipse_cable_geometry(pos: wp.vec3, num_elements=48, twisting_angle=0.0):
    """Generates elliptical parallel-transported orientations directly from only_belt.py"""
    num_points = num_elements + 1
    points = []
    a = 0.248 / 2.0
    b = 0.168 / 2.0
    for i in range(num_points):
        theta = 2.0 * np.pi * i / num_elements
        x = a * np.cos(theta)
        y = b * np.sin(theta)
        points.append(pos + wp.vec3(x, y, 0.0))

    edge_q = newton.utils.create_parallel_transport_cable_quaternions(points, twist_total=float(twisting_angle))
    return points, edge_q


# Main function
class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.sim_substeps = 10
        self.sim_iterations = 5
        self.update_step_interval = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.frame_id = 0
        self.debug_belt_positions = True
        self.debug_every_n_frames = 60

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)

        # Set default builder dynamics parameters
        builder.default_shape_cfg.ke = 1.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.mu = 1.5

        add_table(builder)
        add_board(builder)
        add_pulleys_from_xacro_poses(builder)

        # Create elliptical cable using robust add_rod and transport quaternions
        start_pos = wp.vec3(BELT_CENTER_X, BELT_CENTER_Y, BELT_CENTER_Z)
        cable_points, cable_edge_q = create_ellipse_cable_geometry(
            pos=start_pos,
            num_elements=BELT_NUM_ELEMENTS,
            twisting_angle=0.0,
        )
        
        TOTAL_BELT_MASS = 0.022  # 22 grams in kg
        # Calculate mass per element segment 
        element_mass = TOTAL_BELT_MASS / BELT_NUM_ELEMENTS

        # Use soft elastic parameters
        rod_bodies, _rod_joints = builder.add_rod(
            positions=cable_points,
            quaternions=cable_edge_q,
            radius=BELT_RADIUS,
            stretch_stiffness=5.0e3,
            bend_stiffness=1.5e-1,
            bend_damping=1.0e-1,
            closed=True,
            body_frame_origin="com",
            label="flexible_ellipse_cable",
        )
        self.belt_bodies = list(rod_bodies)

        builder.add_ground_plane()
        builder.color()

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.sim_iterations,
            rigid_avbd_contact_alpha=0.0,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)

        if hasattr(self.viewer, "set_picking_linear_only_bodies"):
            self.viewer.set_picking_linear_only_bodies(self.belt_bodies)

        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(0.20, -0.90, 0.95), -18.0, -32.0)

        self.capture()

    def capture(self):
        if self.solver.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        """Simulate utilizing the exact collision update gating from only_belt.py"""
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            # Gating collision checks avoids excessive constraint collision fighting 
            refresh_contacts = (substep % self.update_step_interval) == 0
            if refresh_contacts:
                self.model.collide(self.state_0, self.contacts)

            self.solver.set_rigid_history_update(refresh_contacts)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.frame_id += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)

        if (
            self.debug_belt_positions
            and self.state_0.body_q is not None
            and len(self.belt_bodies) > 0
            and self.frame_id % self.debug_every_n_frames == 0
        ):
            body_q = self.state_0.body_q.numpy()
            belt_xyz = body_q[np.asarray(self.belt_bodies, dtype=np.int32), :3]
            print("[BELT DEBUG] min xyz =", belt_xyz.min(axis=0), "max xyz =", belt_xyz.max(axis=0))

        self.viewer.end_frame()

    def test_final(self):
        if self.state_0.body_q is not None:
            body_q = self.state_0.body_q.numpy()
            assert np.isfinite(body_q).all(), "Non-finite body transforms"
            belt_xyz = body_q[np.asarray(self.belt_bodies, dtype=np.int32), :3]
            belt_min_z = float(np.min(belt_xyz[:, 2]))
            assert belt_min_z > TABLE_TOP_Z - 0.05, (
                f"Belt fell too far below table: min_z={belt_min_z:.4f}, table_top={TABLE_TOP_Z:.4f}"
            )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)