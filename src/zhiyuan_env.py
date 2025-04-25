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
        self.sensor_names = {
            'orientation': 'body-orientation',
            'angular_vel': 'body-angular-velocity',
            'linear_vel': 'body-linear-vel',
            'linear_accel': 'body-linear-acceleration',
            'left_foot_force': 'left_foot_force',
            'right_foot_force': 'right_foot_force'
        }

        obs_size = 14 + 14 + 3 + 4 + 3 +3 +3 + 2 + 3 + 3  # 关节位置、速度、质心、姿态、线速度 角速度3，加速度3，接触状态2 足底力传感器3*2
        
        self._prev_action = np.zeros(self.action_space.shape[0])  # 初始化为零向量
        
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
        self.max_tilt_angle = np.deg2rad(75)

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
        self.current_contact = np.zeros(2, dtype = bool)
        
        mujoco.mj_step(self.model, self.data)

        self._prev_action = np.zeros(self.action_space.shape[0])
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
        try:
            ang_vel = self.data.sensor(self.sensor_names['angular_vel']).data.copy()
            accel = self.data.sensor(self.sensor_names['linear_accel']).data.copy()
            left_foot_force = self.data.sensor(self.sensor_names['left_foot_force']).data.copy()
            right_foot_force = self.data.sensor(self.sensor_names['right_foot_force']).data.copy()
            #print(f'right-foot-force:{right_foot_force}')
            #print(f'angular-velocity:{ang_vel}\nacceleration{accel}')
        except KeyError:
            # 初始状态可能未激活，填充零值
            ang_vel = np.zeros(3)
            accel = np.zeros(3)
            left_foot_force = np.zeros(3)
            right_foot_force = np.zeros(3)
            #print('sensor error, zeroes')     
        
        contact_state = self.current_contact.astype(np.float32)
        #print(f'contact state:{contact_state}')
        
        qpos = np.array([self.data.qpos[idx] for idx in self.joint_indices])
        qvel = np.array([self.data.qvel[idx] for idx in self.joint_vel_indices])
        com_pos = self.data.subtree_com[self.model.body('x1-body').id].copy()
        body_quat = self.data.sensor('body-orientation').data.copy()
        body_vel = self.data.sensor('body-linear-vel').data.copy()
        return np.concatenate([qpos, qvel, com_pos, body_quat, body_vel,
                               ang_vel, accel, contact_state, left_foot_force, right_foot_force]).astype(np.float32)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        # 计算当前接触力
# 获取地面 geom id
        floor_id = self.model.geom('floor').id
        
        # 获取脚部的 sphere geom id
        left_foot_geom_ids = [self.model.geom(f'left_ankle_roll_sphere_{i}').id for i in range(4)]
        right_foot_geom_ids = [self.model.geom(f'right_ankle_roll_sphere_{i}').id for i in range(4)]
        
        # 检测左脚和右脚是否有接触
        left_contact = False
        right_contact = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2
            if (geom1 == floor_id and geom2 in left_foot_geom_ids) or (geom1 in left_foot_geom_ids and geom2 == floor_id):
                left_contact = True
            if (geom1 == floor_id and geom2 in right_foot_geom_ids) or (geom1 in right_foot_geom_ids and geom2 == floor_id):
                right_contact = True
        
        current_contact = np.array([left_contact, right_contact], dtype=bool)
        self.current_contact = current_contact.copy()
        #print(current_contact)
        contact_freq = np.sum(current_contact ^ self._prev_contact)
        self._prev_contact = current_contact.copy()

        # 计算奖励
        reward = self.compute_reward(current_contact)  # 将contact_forces传递给奖励函数
        terminated = self._check_termination()
        truncated = self._steps >= self.max_episode_steps
        self._steps += 1
        obs = self._get_obs()

        info = {"com_z": self.data.subtree_com[self.model.body('x1-body').id][2], "reward": reward}
        return obs, reward, terminated, truncated, info

    def compute_reward(self, current_contact):
        # 基础物理量获取
        com_vel = self.data.sensor('body-linear-vel').data[0]  # 只关注前进方向速度(x轴)
        ang_vel = np.linalg.norm(self.data.sensor('body-angular-velocity').data)
        accel = np.linalg.norm(self.data.sensor('body-linear-acceleration').data)
        
        # 姿态角计算（使用四元数转欧拉角）
        quat = self.data.sensor('body-orientation').data[[1,2,3,0]]  # Mujoco四元数顺序为(w,x,y,z)
        pitch = np.arcsin(2*(quat[0]*quat[2] - quat[3]*quat[1]))  # 俯仰角
        
        # === 核心奖励项 ===
        # 1. 速度控制奖励（指数衰减+方向引导）
        vel_reward = np.exp(-5*abs(com_vel)) + 0.5*np.exp(-20*(com_vel + 0.3)**2)
        
        # 2. 动态稳定性奖励（角速度惩罚 + 加速度平滑）
        stability_reward = np.exp(-0.5*ang_vel) - 0.01*accel
        
        # 3. 接触优化奖励
        contact_reward = 2.0 if np.all(current_contact) else (
            0.5 if np.any(current_contact) else -1.0
        )
        contact_change_penalty = -0.3 * np.sum(current_contact != self._prev_contact)
        
        # 4. 姿态保持奖励（分阶段惩罚）
        if com_vel > 0.5:  # 高速阶段允许小幅前倾
            pitch_target = np.deg2rad(10)
            pitch_reward = np.exp(-10*abs(pitch - pitch_target))
        else:              # 低速/停止阶段要求直立
            pitch_reward = np.exp(-20*abs(pitch)) 
        
        # 5. 侧向稳定性惩罚（使用横滚角）
        roll = np.arcsin(2*(quat[0]*quat[3] + quat[1]*quat[2]))
        roll_penalty = -0.8 * abs(roll)**1.5  # 非线性惩罚
        
        # 6. 动作平滑性惩罚
        if not hasattr(self, '_prev_action') or self._prev_action is None:
            action_diff = 0.0
        else:
            action_diff = np.linalg.norm(self.data.ctrl - self._prev_action)
        smooth_penalty = -0.002 * action_diff / self.dt
        
        # === 奖励合成 ===
        total_reward = (
            6.0 * vel_reward +
            4.0 * stability_reward +
            3.0 * contact_reward +
            2.5 * pitch_reward +
            1.0 * contact_change_penalty +
            1.5 * roll_penalty +
            1.2 * smooth_penalty
        )
        
        # 历史状态保存
        self._prev_action = self.data.ctrl.copy()
        
        return total_reward

    def _check_termination(self):
        # 获取关键状态量
        com_z = self.data.subtree_com[self.model.body('x1-body').id][2]
        quat = self.data.sensor('body-orientation').data[[1,2,3,0]]  # Mujoco四元数顺序为(w,x,y,z)
        pitch = np.arcsin(2*(quat[0]*quat[2] - quat[3]*quat[1]))  # 俯仰角
        roll = np.arcsin(2*(quat[0]*quat[3] + quat[1]*quat[2]))
        # 动态阈值调整
        vel = abs(self.data.qvel[0])
        max_pitch = np.deg2rad(30) if vel > 0.5 else np.deg2rad(15)
        
        termination_conditions = (
            com_z < 0.4 or
            abs(pitch) > max_pitch or
            abs(roll) > np.deg2rad(25) or
            (vel < 0.1 and self._steps - self._stable_steps > 100)
        )
        
        # 更新稳定计时
        if vel < 0.05 and abs(pitch) < np.deg2rad(5):
            self._stable_steps += 1
        else:
            self._stable_steps = 0
            
        return termination_conditions

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
