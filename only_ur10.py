import argparse
import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton import JointTargetMode


def as_numpy(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def quat_to_vec4(q):
    return wp.vec4(float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def parse_indices(text):
    if text is None or text.strip() == "":
        return []
    return [int(v.strip()) for v in text.split(",") if v.strip() != ""]


def find_body_index(model, name_hint):
    labels = getattr(model, "body_label", [])

    if name_hint:
        name_hint = name_hint.lower()
        for i, label in enumerate(labels):
            if name_hint in label.lower():
                return i

    keywords = [
        "ee_link",
        "tool0",
        "wrist_3_link",
    ]

    for key in keywords:
        for i, label in enumerate(labels):
            if key in label.lower():
                return i

    return model.body_count - 1


def print_model_tree(model):
    print("\n========== BODY LABELS ==========")
    for i, name in enumerate(getattr(model, "body_label", [])):
        print(f"body {i:3d}: {name}")

    print("\n========== JOINT LABELS / COORD INDICES ==========")
    joint_labels = getattr(model, "joint_label", [])
    q_start = as_numpy(model.joint_q_start)
    dof_dim = as_numpy(model.joint_dof_dim)

    for j, name in enumerate(joint_labels):
        dofs = int(dof_dim[j, 0] + dof_dim[j, 1])
        q0 = int(q_start[j])
        q_indices = list(range(q0, q0 + dofs))
        print(f"joint {j:3d}: q={q_indices}  name={name}")

    print("=================================\n")


@wp.kernel
def set_gripper_target_kernel(
    joint_target_q: wp.array2d[wp.float32],
    gripper_indices: wp.array[wp.int32],
    gripper_values: wp.array[wp.float32],
):
    i = wp.tid()
    q_idx = gripper_indices[i]
    joint_target_q[0, q_idx] = gripper_values[i]


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True

        self.viewer = viewer
        self.device = wp.get_device()

        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.arm_dofs = 6

        # Build robot
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        height = 1.2
        base_tf = wp.transform(
            wp.vec3(0.0, 0.0, height),
            wp.quat_identity(),
        )

        if args.asset_file:
            if args.asset_type == "usd":
                builder.add_usd(
                    args.asset_file,
                    xform=base_tf,
                    collapse_fixed_joints=False,
                    enable_self_collisions=False,
                    hide_collision_shapes=False,
                )
            elif args.asset_type == "urdf":
                builder.add_urdf(
                    args.asset_file,
                    xform=base_tf,
                    floating=False,
                    enable_self_collisions=False,
                    parse_visuals_as_colliders=False,
                )
            else:
                raise ValueError(f"Unknown asset type: {args.asset_type}")

        else:
            # Load stock UR10 arm
            asset_path = newton.utils.download_asset("universal_robots_ur10")
            asset_file = str(asset_path / "usd" / "ur10_instanceable.usda")

            builder.add_usd(
                asset_file,
                xform=base_tf,
                collapse_fixed_joints=False,
                enable_self_collisions=False,
                hide_collision_shapes=True,
            )

            # CENTERED ROBOTIQ 2F-85 GRIPPER MOUNT
            # For the stock UR10 USD, body 7 is usually /ur10/ee_link.
            tool_body_idx = 7
            print(f"[INFO] Attaching 2f85.xml centered to body {tool_body_idx}.")

            gripper_pos = wp.vec3(0.0, 0.0, 0.0)

            # This keeps the gripper aligned with the UR10 tool frame.
            gripper_rot = wp.quat_from_axis_angle(
                wp.vec3(0.0, 1.0, 0.0),
                np.pi / 2.0,
            )

            gripper_offset_tf = wp.transform(gripper_pos, gripper_rot)

            builder.add_mjcf(
                "2f85.xml",
                parent_body=tool_body_idx,
                xform=gripper_offset_tf,
            )

        builder.add_shape_cylinder(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, height / 2.0)),
            half_height=height / 2.0,
            radius=0.08,
        )

        builder.add_ground_plane()

        # Joint gains
        for i in range(len(builder.joint_target_ke)):
            if i < self.arm_dofs:
                builder.joint_target_ke[i] = 1000.0
                builder.joint_target_kd[i] = 100.0
            else:
                # Strong but not excessive gripper tracking.
                # Too high can increase jitter/penetration during contact.
                builder.joint_target_ke[i] = 200.0
                builder.joint_target_kd[i] = 15.0

            builder.joint_target_mode[i] = int(JointTargetMode.POSITION)

        self.model = builder.finalize()

        if args.print_model:
            print_model_tree(self.model)

        # Initial arm pose
        initial_joints = np.zeros(self.model.joint_coord_count, dtype=np.float32)

        initial_joints[0] = 0.0
        initial_joints[1] = -np.pi / 2.0
        initial_joints[2] = np.pi / 2.0
        initial_joints[3] = -np.pi / 2.0
        initial_joints[4] = -np.pi / 2.0
        initial_joints[5] = 0.0

        self.model.joint_q.assign(initial_joints)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            disable_contacts=False,
        )

        self.viewer.set_model(self.model)

        newton.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            self.state_0,
        )

        # End-effector IK target
        self.ee_index = find_body_index(self.model, args.ee_body)

        print(f"[INFO] IK end-effector body index: {self.ee_index}")
        if hasattr(self.model, "body_label"):
            print(f"[INFO] IK end-effector body label: {self.model.body_label[self.ee_index]}")

        body_q_np = self.state_0.body_q.numpy()
        ee_tf = wp.transform(*body_q_np[self.ee_index])
        ee_pos = wp.transform_get_translation(ee_tf)
        ee_rot = wp.transform_get_rotation(ee_tf)

        self.ee_pos_target = wp.array(
            [ee_pos],
            dtype=wp.vec3,
            device=self.device,
        )

        self.ee_rot_target = wp.array(
            [quat_to_vec4(ee_rot)],
            dtype=wp.vec4,
            device=self.device,
        )

        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=self.ee_pos_target,
        )

        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=self.ee_rot_target,
        )

        self.joint_limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0,
        )

        self.n_coords = self.model.joint_coord_count
        self.joint_q_ik = wp.clone(
            self.model.joint_q.reshape((1, self.n_coords))
        )

        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[
                self.pos_obj,
                self.rot_obj,
                self.joint_limit_obj,
            ],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        self.ik_iters = 32

        # Gripper joint discovery
        gripper_args = args.gripper_indices

        if not gripper_args and not args.asset_file:
            driver_indices = []
            fallback_finger_indices = []

            joint_labels = getattr(self.model, "joint_label", [])
            q_start = as_numpy(self.model.joint_q_start)

            for j, name in enumerate(joint_labels):
                lname = name.lower()

                # Prefer actual active driver joints.
                if "driver_joint" in lname:
                    driver_indices.append(int(q_start[j]))

                # Fallback only. Avoid driving every passive internal linkage.
                elif lname.endswith("finger_joint") or "/finger_joint" in lname:
                    fallback_finger_indices.append(int(q_start[j]))

            if driver_indices:
                g_indices = driver_indices
            else:
                g_indices = fallback_finger_indices

            if g_indices:
                gripper_args = ",".join(map(str, sorted(list(set(g_indices)))))

        self.gripper_coord_indices = parse_indices(gripper_args)

        print(f"[INFO] Driving gripper coordinate indices: {self.gripper_coord_indices}")

        self.gripper_indices_wp = wp.array(
            self.gripper_coord_indices,
            dtype=wp.int32,
            device=self.device,
        )

        # Safe full open / full close targets
        lower_np = as_numpy(self.model.joint_limit_lower)
        upper_np = as_numpy(self.model.joint_limit_upper)

        self.gripper_open_values = []
        self.gripper_closed_values = []

        limit_margin = 1e-4

        for q_idx in self.gripper_coord_indices:
            lo = float(lower_np[q_idx])
            hi = float(upper_np[q_idx])

            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                open_q = lo + limit_margin
                closed_q = hi - limit_margin
            else:
                # Fallback only if the gripper joint limits are missing.
                open_q = float(args.gripper_open)
                closed_q = float(args.gripper_closed)

            self.gripper_open_values.append(open_q)
            self.gripper_closed_values.append(closed_q)

        print(f"[INFO] Gripper open targets:   {self.gripper_open_values}")
        print(f"[INFO] Gripper closed targets: {self.gripper_closed_values}")

        self.gripper_open_values_wp = wp.array(
            self.gripper_open_values,
            dtype=wp.float32,
            device=self.device,
        )

        self.gripper_closed_values_wp = wp.array(
            self.gripper_closed_values,
            dtype=wp.float32,
            device=self.device,
        )

        self.gripper_is_closed = False

        self.joint_target_q_view = self.control.joint_target_q.reshape(
            (1, self.n_coords)
        )

    def set_end_effector_target(self, pos, quat_xyzw):
        pos_arr = wp.array(
            [wp.vec3(*pos)],
            dtype=wp.vec3,
            device=self.device,
        )

        rot_arr = wp.array(
            [wp.vec4(*quat_xyzw)],
            dtype=wp.vec4,
            device=self.device,
        )

        self.ee_pos_target = pos_arr
        self.ee_rot_target = rot_arr

        self.pos_obj.set_target_positions(self.ee_pos_target)
        self.rot_obj.set_target_rotations(self.ee_rot_target)

    def open_gripper(self):
        self.gripper_is_closed = False

    def close_gripper(self):
        self.gripper_is_closed = True

    def solve_ik_and_set_targets(self):
        self.ik_solver.reset()

        self.ik_solver.step(
            self.joint_q_ik,
            self.joint_q_ik,
            iterations=self.ik_iters,
        )

        # Drive only the UR10 arm joints from IK.
        wp.copy(
            dest=self.joint_target_q_view[:, : self.arm_dofs],
            src=self.joint_q_ik[:, : self.arm_dofs],
        )

        # Drive gripper independently using safe open/closed joint limits.
        if len(self.gripper_coord_indices) > 0:
            if self.gripper_is_closed:
                target_values = self.gripper_closed_values_wp
            else:
                target_values = self.gripper_open_values_wp

            wp.launch(
                set_gripper_target_kernel,
                dim=len(self.gripper_coord_indices),
                inputs=[
                    self.joint_target_q_view,
                    self.gripper_indices_wp,
                    target_values,
                ],
                device=self.device,
            )

    def simulate(self):
        self.solve_ik_and_set_targets()

        self.model.collide(self.state_0, self.contacts)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # Circular end-effector motion
        center = np.array([0.55, 0.0, 1.25])
        radius = 0.12

        target_pos = np.array(
            [
                center[0] + radius * np.cos(self.sim_time),
                center[1] + radius * np.sin(self.sim_time),
                center[2],
            ],
            dtype=np.float32,
        )

        target_quat_xyzw = (0.0, 0.7071, 0.0, 0.7071)

        self.set_end_effector_target(
            target_pos,
            target_quat_xyzw,
        )

        if int(self.sim_time) % 4 < 2:
            self.open_gripper()
        else:
            self.close_gripper()

        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        pass

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()

        parser.add_argument("--asset-file", type=str, default="")
        parser.add_argument(
            "--asset-type",
            type=str,
            default="usd",
            choices=["usd", "urdf"],
        )

        parser.add_argument("--ee-body", type=str, default="ee_link")

        # Leave empty for automatic gripper driver-joint discovery.
        parser.add_argument("--gripper-indices", type=str, default="")

        # These are only fallback values if the MJCF joint limits are unavailable.
        parser.add_argument("--gripper-open", type=float, default=0.0)
        parser.add_argument("--gripper-closed", type=float, default=0.8)

        parser.add_argument("--print-model", action="store_true")

        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)