import os
import torch
import torch.nn as nn
import torch.onnx
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from zhiyuan_env import ZhiyuanEnv
import argparse
import mujoco
import numpy as np
from datetime import datetime
from stable_baselines3.common.vec_env import DummyVecEnv

class CustomPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.net(observations)

def export_to_onnx(model, output_path, input_shape):
    dummy_input = torch.randn(1, input_shape).to(model.device)
    torch.onnx.export(
        model.policy.actor.net,
        dummy_input,
        output_path,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"ONNX 模型保存至: {output_path}")

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", default="../checkpoints/ppo_checkpoint.zip")
    args = parser.parse_args()

    xml_path = "../models/zhiyuan_scene.xml"
    dataset_path = "../data/walking_dataset.csv"
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    onnx_path = f"../policy/stop_policy_{start_time}.onnx"
    num_envs = 16

    # 创建网格位置（32x32）
    grid_size = int(np.ceil(np.sqrt(num_envs)))
    grid_positions = [(i % grid_size, i // grid_size) for i in range(num_envs)]


    def make_env(robot_id, grid_pos):
        def _init():
            return ZhiyuanEnv(
                xml_path=xml_path,
                dataset_path=dataset_path,
                robot_id=robot_id,
                render_mode="human",
                grid_pos=grid_pos
            )
        return _init

    # 直接构造DummyVecEnv
    env = DummyVecEnv([make_env(i, grid_positions[i]) for i in range(num_envs)])
    # 添加这里
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path="../checkpoints/",
        name_prefix="ppo_checkpoint"
    )
    # 自定义策略网络
    policy_kwargs = {
        "features_extractor_class": CustomPolicy,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": [dict(pi=[128, 64], vf=[128, 64])]
    }

    # 加载或新建模型
    if os.path.exists(args.checkpoint_path):
        print(f"加载断点: {args.checkpoint_path}")
        model = PPO.load(args.checkpoint_path, env=env, custom_objects={"policy_kwargs": policy_kwargs})
    else:
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            n_steps=128,
            batch_size=256,
            n_epochs=10,
            learning_rate=3e-4,
            clip_range=0.2,
            verbose=1,
            tensorboard_log="../logs/"
        )

    # 训练并渲染
    total_timesteps = 5000 * 128
    timesteps_per_iter = 128
    viewer = None
    render_context = None
    model_xml = mujoco.MjModel.from_xml_path(xml_path)

    obs = env.reset() 
    
    for i in range(5000):
        model.learn(
            total_timesteps=timesteps_per_iter,
            callback=checkpoint_callback,
            reset_num_timesteps=False
        )
        action, _states = model.predict(obs)  # 从模型中获取动作
        # 获取当前奖励
        obs, reward, terminated, truncated = model.env.step(action)  # 直接从step中获取奖励
        avg_reward = np.mean(reward)
        print(f"Step {i}: Current Reward = {avg_reward}")  # 打印奖励

        # 渲染
        if env.get_attr("render_mode")[0] == "rgb_array":
            width, height = 1280, 720
            framebuffer = np.zeros((height, width, 3), dtype=np.uint8)

            # 渲染每个环境的窗口
            if render_context is None:
                render_context = mujoco.MjrContext(model_xml, mujoco.mjtFontScale.mjFONTSCALE_150)

            if viewer is None:
                viewer = mujoco.viewer.launch_passive(model_xml, env.get_attr("data")[0])

            for j in range(num_envs):
                data = env.get_attr("data")[j]
                row, col = grid_positions[j]
                viewport = mujoco.MjrRect(col * width // grid_size, (grid_size - 1 - row) * height // grid_size, width // grid_size, height // grid_size)
                
                # 设置相机和场景
                cam = mujoco.MjvCamera()
                cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                cam.fixedcamid = model_xml.camera('track').id
                scene = mujoco.MjvScene(model_xml, maxgeom=10000)

                mujoco.mjv_updateScene(model_xml, data, mujoco.MjvOption(), None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
                mujoco.mjr_render(viewport, scene, render_context)
                mujoco.mjr_readPixels(framebuffer, None, viewport, render_context)

            viewer.sync()
            mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height), viewer.user_scn, render_context)

    model.save("../checkpoints/ppo_final.zip")
    export_to_onnx(model, onnx_path, env.observation_space.shape[0])

    if viewer is not None:
        viewer.close()



if __name__ == "__main__":
    train()