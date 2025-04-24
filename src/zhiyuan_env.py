import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import os
import pandas as pd
import torch

class ZhiyuanEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, xml_path: str, dataset_path: str, robot_id: int = 0, render_mode: str = None, grid_pos: tuple = (0, 0)):
        super().__init__()
        xml_path = os.path.abspath(xml_path)
        dataset_path = os.path.abspath(dataset_path)
        meshes_dir = os.path.abspath(os.path.join(os.path.dirname(xml_path), "..", "meshes"))
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML 文件未找到: {xml_path}")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集未找到: {dataset_path}")
        if not os.path.exists(meshes_dir):
            raise FileNotFoundError(f"网格目录未找到: {meshes_dir}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.robot_id = robot_id
        self.grid_pos = grid_pos  # 用于渲染时的网格位置
        self.render_mode = render_mode
        self.viewer = None
        self.render_context = None
        self._prev_contact = None

        # 动作空间（29个执行器）
        n_actuators = 29
        actuator_low = np.array([-150, -150, -150, -150, -150, -150, -150, -150, -150, -150, -150, -150, -150, -150,
                                 -150, -150, -150, -150, -50, -50, -150, -18, -18, -150, -50, -50, -150, -18, -18])
        actuator_high = np.array([150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150,
                                  150, 150, 150, 150, 50, 50, 150, 18, 18, 150, 50, 50, 150, 18, 18])
        self.action_space = spaces.Box(
            low=actuator_low,
            high=actuator_high,
            shape=(n_actuators,),
            dtype=np.float32
        )

        obs_size = 14 + 14 + 3 + 4 + 3  # 关节位置、速度、质心、姿态、线速度
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
                
        #print(f"Observation space shape: {self.observation_space.shape}") 
        # 应为 (14+14+3+4+3=38,) 即 (38,)    

        self.joint_names = [
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
            'left_knee_pitch_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'left_shoulder_pitch_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
            'right_knee_pitch_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
            'right_shoulder_pitch_joint'
        ]
        self.joint_indices = [self.model.joint(name).qposadr[0] for name in self.joint_names]
        self.joint_vel_indices = [self.model.joint(name).dofadr[0] for name in self.joint_names]

        self.dataset_path = dataset_path
        self._load_dataset()

        self.frame_skip = 1
        self.dt = self.model.opt.timestep * self.frame_skip
        self.max_episode_steps = 1000
        self._steps = 0

        self.target_com_z = 0.6
        self.stable_threshold = 0.05
        self.max_tilt_angle = np.deg2rad(70)

        self.feet_indices = [
            self.model.geom('left_ankle_roll').id,
            self.model.geom('right_ankle_roll').id
        
        ]

    def _load_dataset(self):
        df = pd.read_csv(self.dataset_path)
        pos_diff_cols = [f"{name}_pos_diff" for name in self.joint_names]
        vel_cols = [f"{name}_vel" for name in self.joint_names]
        self.dataset_pos_diff = df[pos_diff_cols].values
        self.dataset_vel = df[vel_cols].values
        self.dataset_base_ang_vel = df[['base_ang_vel_x', 'base_ang_vel_y', 'base_ang_vel_z']].values
        self.dataset_base_euler = df[['base_euler_x', 'base_euler_y', 'base_euler_z']].values
        self.dataset_size = len(df)

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        idx = np.random.randint(0, self.dataset_size)
        pos_diff = self.dataset_pos_diff[idx]
        vel = self.dataset_vel[idx]
        base_ang_vel = self.dataset_base_ang_vel[idx]
        base_euler = self.dataset_base_euler[idx]

        max_initial_tilt = np.deg2rad(45)
        attempts = 0
        while attempts < 10:
            quat = self._euler_to_quat(base_euler)
            tilt_error = 2 * np.arccos(np.abs(quat[0]))
            if tilt_error <= max_initial_tilt:
                break
            idx = np.random.randint(0, self.dataset_size)
            pos_diff = self.dataset_pos_diff[idx]
            vel = self.dataset_vel[idx]
            base_ang_vel = self.dataset_base_ang_vel[idx]
            base_euler = self.dataset_base_euler[idx]
            attempts += 1

        # 设置初始位置（基于网格位置偏移）
        grid_x, grid_y = self.grid_pos
        self.data.qpos[:3] = [grid_x * 1.5, grid_y * 1.5, 0.6]  # 网格间距1.5
        self.data.qpos[3:7] = quat
        home_qpos = self.model.key_qpos[self.model.keyframe('home_default').id]
        self.data.qpos[7:] = home_qpos[7:].copy()
        for j, joint_idx in enumerate(self.joint_indices):
            self.data.qpos[joint_idx] = home_qpos[joint_idx] + np.clip(
                pos_diff[j],
                self.model.jnt_range[self.model.joint(self.joint_names[j]).id][0] - home_qpos[joint_idx],
                self.model.jnt_range[self.model.joint(self.joint_names[j]).id][1] - home_qpos[joint_idx]
            )

        self.data.qvel[:3] = [1.0, 0, 0]
        self.data.qvel[3:6] = base_ang_vel
        for j, joint_idx in enumerate(self.joint_vel_indices):
            self.data.qvel[joint_idx] = vel[j]

        self.data.qpos[7:] += np.random.uniform(-0.05, 0.05, size=self.data.qpos[7:].shape)
        self.data.qvel[6:] += np.random.uniform(-0.1, 0.1, size=self.data.qvel[6:].shape)

        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        
        self._prev_contact = np.zeros(2, dtype=bool)

        return self._get_obs(), {}

    def _euler_to_quat(self, euler):
        roll, pitch, yaw = euler
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        q = np.empty(4)
        q[0] = cr * cp * cy + sr * sp * sy
        q[1] = sr * cp * cy - cr * sp * sy
        q[2] = cr * sp * cy + sr * cp * sy
        q[3] = cr * cp * sy - sr * sp * cy
        return q

    def _get_obs(self):
        qpos = np.array([self.data.qpos[idx] for idx in self.joint_indices])
        qvel = np.array([self.data.qvel[idx] for idx in self.joint_vel_indices])
        com_pos = self.data.subtree_com[self.model.body('x1-body').id].copy()
        body_quat = self.data.sensor('body-orientation').data.copy()
        body_vel = self.data.sensor('body-linear-vel').data.copy()
        return np.concatenate([qpos, qvel, com_pos, body_quat, body_vel]).astype(np.float32)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        # 计算当前接触力
        contact_forces = np.zeros(2, dtype=np.float32)
        for j, body_name in enumerate(['link_left_ankle_roll', 'link_right_ankle_roll']):
            body_id = self.model.body(body_name).id
            geom_start = self.model.body_geomadr[body_id]
            geom_num = self.model.body_geomnum[body_id]
            for k in range(geom_start, geom_start + geom_num):
                geom_type = self.model.geom(k).type
                geom_contype = self.model.geom(k).contype
                if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE and geom_contype == 2:
                    contact_forces[j] = self.data.cfrc_ext[k][2]
                    break
        
        current_contact = contact_forces > 1.0
        contact_freq = np.sum(current_contact ^ self._prev_contact) 
        self._prev_contact = current_contact.copy()

        # 计算奖励
        reward = self.compute_reward(contact_forces,current_contact)  # 将contact_forces传递给奖励函数
        terminated = self._check_termination()
        truncated = self._steps >= self.max_episode_steps
        self._steps += 1
        obs = self._get_obs()

        info = {"com_z": self.data.subtree_com[self.model.body('x1-body').id][2], "reward": reward}
        return obs, reward, terminated, truncated, info

    def compute_reward(self,contact_forces,current_contact):
    # 质心高度奖励（目标高度0.7米）
        com_z = self.data.subtree_com[self.model.body('x1-body').id][2]
        com_reward = np.exp(-5 * abs(com_z - 0.6))

        # 质心速度惩罚（鼓励速度归零）
        com_vel = np.linalg.norm(self.data.sensor('body-linear-vel').data)
        com_vel_penalty = -2 * com_vel

        # 姿态稳定性奖励（鼓励直立）
        body_quat = np.array(self.data.sensor('body-orientation').data, dtype=np.float32)
        quat_error = 1 - np.inner(body_quat, np.array([1, 0, 0, 0], dtype=np.float32))**2
        quat_reward = np.exp(-5 * quat_error)

        # 双脚接触奖励
        contact_forces = np.zeros(2, dtype=np.float32)
        for j, body_name in enumerate(['link_left_ankle_roll', 'link_right_ankle_roll']):
            body_id = self.model.body(body_name).id
            geom_start = self.model.body_geomadr[body_id]
            geom_num = self.model.body_geomnum[body_id]
            for k in range(geom_start, geom_start + geom_num):
                if self.model.geom(k).type == mujoco.mjtGeom.mjGEOM_SPHERE and self.model.geom(k).contype == 2:
                    contact_forces[j] = np.array(self.data.cfrc_ext[k][2], dtype=np.float32)
                    break
        contact = contact_forces > 1.0
        if np.all(contact):
            contact_reward = 1.0  # 两脚着地
        elif np.any(contact):
            contact_reward = 0.5  # 单脚着地
        else:
            contact_reward = -0.5  # 两脚不着地

        # 脚部速度惩罚（鼓励静止）
        feet_vel = np.zeros((2, 3), dtype=np.float32)
        for j, body_name in enumerate(['link_left_ankle_roll', 'link_right_ankle_roll']):
            body_id = self.model.body(body_name).id
            geom_start = self.model.body_geomadr[body_id]
            geom_num = self.model.body_geomnum[body_id]
            for k in range(geom_start, geom_start + geom_num):
                if self.model.geom(k).type == mujoco.mjtGeom.mjGEOM_SPHERE and self.model.geom(k).contype == 2:
                    feet_vel[j] = np.array(self.data.geom_xvelp[k][:3], dtype=np.float32)
                    break
        feet_vel_penalty = -np.sum(np.square(feet_vel))

        # 髋关节位置惩罚（鼓励默认站立姿势）
        hip_indices = [self.model.joint(name).qposadr[0] for name in ['left_hip_roll_joint', 'left_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint']]
        hip_pos = np.array([self.data.qpos[i] for i in hip_indices], dtype=np.float32)
        hip_penalty = -np.sum(np.square(hip_pos))

        # 动作平滑惩罚
        action_penalty = -0.001 * np.sum(np.square(np.array(self.data.ctrl, dtype=np.float32)))
        
        # 姿态倾斜奖励（鼓励站直/略向前弯腰，惩罚后倒）
        pitch_angle = np.arcsin(2 * (body_quat[0] * body_quat[2] - body_quat[1] * body_quat[3]))  # 提取俯仰角（pitch）
        pitch_reward = np.exp(-5 * (pitch_angle - 0.1)**2) - 0.5 * np.clip(pitch_angle, 0, np.pi/4)  # 目标略前倾0.1弧度，惩罚后倒

        forward_vel = self.data.sensor('body-linear-vel').data[0]  # x 轴速度
        deceleration_reward = -2 * abs(forward_vel)  # 奖励速度接近 0
        
        contact_freq = np.sum(current_contact ^ self._prev_contact)  # 接触切换频率
        contact_freq_reward = 0.3 * contact_freq  # 奖励频繁接触
        
        # 存活奖励
        alive_reward = 1.0

        # 总奖励（权重初步设计，需调优）
        total_reward = (
            3.0 * com_reward +          # 质心高度
            1.0 * com_vel_penalty +     # 质心速度
            2.0 * quat_reward +         # 姿态
            1.5 * contact_reward +      # 脚部接触
            0.1 * feet_vel_penalty +    # 脚部速度
            0.2 * hip_penalty +         # 髋关节
            0.1 * action_penalty +      # 动作平滑
            10.0 * alive_reward+         # 存活
            1.0 * pitch_reward+
            1.0 * deceleration_reward+
            0.3 * contact_freq_reward
        )

        return total_reward

    def _check_termination(self):
        com_z = self.data.subtree_com[self.model.body('x1-body').id][2]
        body_quat = self.data.sensor('body-orientation').data
        tilt_error = 2 * np.arccos(np.abs(body_quat[0]))
        body_vel = np.linalg.norm(self.data.sensor('body-linear-vel').data)
        if com_z < 0.3 or com_z > 0.9 or tilt_error > self.max_tilt_angle:
            return True
        if (abs(com_z - self.target_com_z) < self.stable_threshold and
                tilt_error < np.deg2rad(5) and
                body_vel < self.stable_threshold):
            return True
        return False

    def render(self):
        if self.render_mode is None:
            return
        if self.render_mode == "rgb_array":
            viewport = mujoco.MjrRect(0, 0, 640, 480)
            rgb = np.zeros((viewport.height, viewport.width, 3), dtype=np.uint8)
            mujoco.mj_render(self.model, self.data, rgb)
            return rgb[::-1, :, :]
        # 修改render方法中human模式的部分：
        elif self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()  # 仅同步数据，不阻塞


    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.render_context is not None:
            self.render_context = None
