import os
import torch
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from zhiyuan_env import ZhiyuanEnv
import mujoco
import time
from train import CustomPolicy


def play():
    # 设置文件路径
    xml_path = "../models/zhiyuan_scene.xml"
    dataset_path = "../data/walking_dataset.csv"
    checkpoint_path = "../checkpoints/best/best_model"
    policy_kwargs = {
    "features_extractor_class": CustomPolicy,
    "features_extractor_kwargs": {"features_dim": 512},  # 根据需要调整
    "net_arch": [dict(pi=[512, 384, 256, 128], vf=[512, 384, 256, 128])] # 可以根据需要修改网络结构
    }   

    # 加载训练好的模型
    print(f"加载断点: {checkpoint_path}")
    custom_objects = {
    "policy_kwargs": policy_kwargs,
    "clip_range": 0.15,  # 与 train.py 中的值一致
    "lr_schedule": lambda _: 3e-4  # 固定学习率，与 train.py 一致
    }
    model = PPO.load(checkpoint_path, env=None, custom_objects=custom_objects)
    
    # 创建仿真环境
    def make_env():
        return ZhiyuanEnv(
            xml_path=xml_path,
            dataset_path=dataset_path,
            render_mode="human",  # 需要设置渲染模式为 "human"
            grid_pos=(0, 0)
        )
    
    env = make_env()
    # 在env = make_env()之后添加：
    viewer = mujoco.viewer.launch_passive(env.model, env.data)

    # 从数据集中随机抽取一组数据作为初始状态
    df = pd.read_csv(dataset_path)
    idx = np.random.randint(0, len(df))
    pos_diff = df.iloc[idx][[f"{name}_pos_diff" for name in env.joint_names]].values
    vel = df.iloc[idx][[f"{name}_vel" for name in env.joint_names]].values
    base_ang_vel = df.iloc[idx][['base_ang_vel_x', 'base_ang_vel_y', 'base_ang_vel_z']].values
    base_euler = df.iloc[idx][['base_euler_x', 'base_euler_y', 'base_euler_z']].values

    # 重置环境并初始化状态
    env.reset(seed=None)
    grid_pos = env.grid_pos
    env.data.qpos[:3] = [grid_pos[0] * 1.5, grid_pos[1] * 1.5, 0.7]  # 网格间距
    quat = env._euler_to_quat(base_euler)
    env.data.qpos[3:7] = quat
    home_qpos = env.model.key_qpos[env.model.keyframe('home_default').id]
    env.data.qpos[7:] = home_qpos[7:].copy()
    
    # 设置初始关节位置、速度
    for j, joint_idx in enumerate(env.joint_indices):
        env.data.qpos[joint_idx] = home_qpos[joint_idx] + np.clip(
            pos_diff[j],
            env.model.jnt_range[env.model.joint(env.joint_names[j]).id][0] - home_qpos[joint_idx],
            env.model.jnt_range[env.model.joint(env.joint_names[j]).id][1] - home_qpos[joint_idx]
        )

    env.data.qvel[:3] = [1.0, 0, 0]
    env.data.qvel[3:6] = base_ang_vel
    for j, joint_idx in enumerate(env.joint_vel_indices):
        env.data.qvel[joint_idx] = vel[j]

    # Forward to initialize the state
    mujoco.mj_forward(env.model, env.data)

    # 获取初始观察状态
    obs = env._get_obs()

    #manual test
    obs = env.reset()[0]  # 获取初始观测
    action, _ = model.predict(obs)
    print(f"Predicted action: {action}")

    # 开始模型预测并运行
    total_steps = 1000  # 运行1000步，展示效果
# 在每个仿真步骤中，先执行一步仿真，然后渲染
    for step in range(total_steps):
        print(f"Executing step {step}")
        action, _ = model.predict(obs)
        
        # 执行动作并获取新状态
        obs, reward, terminated, truncated, info = env.step(action)  # 这行已经包含mujoco.mj_step
        
        # 渲染更新
        viewer.sync()
        
        # 打印信息
        print(f"Step {step}: Reward = {reward}")
        time.sleep(1.0 / env.metadata["render_fps"])
        
        if terminated or truncated:
            break


    # 关闭环境和渲染器
    env.close()
    print("结束仿真!")


if __name__ == "__main__":
    play()
