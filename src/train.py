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
import time

class CustomPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]  # 获取观察空间的维度，应该是38
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 512),  
            nn.ReLU(),
            nn.Linear(512, 256),  # 中间层输出维度是 256
            nn.ReLU(),
            nn.Linear(256, features_dim),  # 最后输出特征维度
            nn.ReLU()
        )

    def forward(self, observations):
        return self.net(observations)




def export_to_onnx(model, output_path, input_shape):
    # 获取完整的观察空间形状
    if isinstance(input_shape, int):
        input_shape = (input_shape,)  # 将整数转换为元组
    
    # 获取策略网络的actor部分
    #actor = model.policy.mlp_extractor.policy_net
    actor = model.policy.mlp_extractor.policy_net if hasattr(model.policy.mlp_extractor, 'policy_net') else model.policy
    
    # 创建正确维度的输入张量
    dummy_input = torch.randn(1, *input_shape)
    
    # 导出actor网络
    torch.onnx.export(
        actor,
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
    # 替换原来的DummyVecEnv创建代码
    env = make_env(0, (0,0))()  # 创建单个环境用于可视化
    eval_env = DummyVecEnv([make_env(i, grid_positions[i]) for i in range(num_envs)])  # 用于训练的多环境
    
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
        model = PPO.load(args.checkpoint_path, env=eval_env, custom_objects={"policy_kwargs": policy_kwargs})
    else:
        model = PPO(
            "MlpPolicy",
            eval_env,
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

    obs = eval_env.reset() 
    viewer = mujoco.viewer.launch_passive(env.model, env.data)
    
    try:
        obs = eval_env.reset()  # 使用 eval_env 初始化 obs

        for i in range(5000):
            # 训练步骤
            model.learn(
                total_timesteps=timesteps_per_iter,
                callback=checkpoint_callback,
                reset_num_timesteps=False
            )
            
            # 每100次迭代进行一次可视化评估
            if i % 100 == 0:
                # 重置可视化环境
                obs = env.reset()[0]
                total_eval_reward = 0
                
                # 运行评估episode（最多200步）
                for eval_step in range(200):
                    action, _ = model.predict(obs)
                    obs, reward, terminated, truncated, _ = env.step(action)  # 单环境返回单个reward
                    total_eval_reward += reward  # 直接累加这个reward值
                    
                    # 更新可视化
                    if hasattr(viewer, 'is_running') and viewer.is_running():  # 确保viewer未关闭
                        viewer.sync()
                    time.sleep(1.0 / env.metadata["render_fps"])
                    
                    if terminated or truncated:
                        break
                        
                print(f"Evaluation after {i} iterations - Total reward: {total_eval_reward:.2f}")
            
            # 正常训练信息打印
            action, _states = model.predict(obs)
            _, rewards, _, _ = eval_env.step(action)  # 多环境返回rewards数组
            avg_reward = np.mean(rewards)  # 计算平均奖励
            print(f"Iteration {i}: Avg reward = {avg_reward:.2f}")

        # 在for i in range(5000):循环结束后添加：
        if 4999 % 100 != 0:  # 如果最后一次迭代不是评估周期
            # 执行最终评估
            obs = env.reset()
            total_eval_reward = 0
            for eval_step in range(200):
                action, _ = model.predict(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_eval_reward += reward
                if hasattr(viewer, 'is_running') and viewer.is_running():
                    viewer.sync()
                if terminated or truncated:
                    break
            print(f"Final evaluation - Total reward: {total_eval_reward:.2f}")
                
    finally:
        # 确保资源释放
        if viewer is not None:
            viewer.close()
        
        # 保存最终模型
        model.save("../checkpoints/ppo_final.zip")
        export_to_onnx(model, onnx_path, env.observation_space.shape)
        print("Training completed and model saved.")


    if viewer is not None:
        viewer.close()

if __name__ == "__main__":
    train()