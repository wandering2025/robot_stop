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
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

class CustomPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        # 确保第一层输入维度匹配观察空间
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 512),  # 这里必须是观察空间维度
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.Tanh()
        )

    def forward(self, observations):
        return self.net(observations)

# 修改文件: `train.py` 中的 export_to_onnx 函数
def export_to_onnx(model, output_path, input_shape):
    device = next(model.policy.parameters()).device
    
    # 创建正确维度的输入（确保与观察空间匹配）
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    # 导出完整的actor网络（包含特征提取器）
    model.policy.eval()  # 确保在eval模式
    
    torch.onnx.export(
        model.policy,
        dummy_input,
        output_path,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

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
    env = make_env(0, (0,0))()  # 创建单个环境用于可视化
    eval_env = DummyVecEnv([make_env(i, grid_positions[i]) for i in range(num_envs)])  # 用于训练的多环境
    
    # 添加检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path="../checkpoints/",
        name_prefix="ppo_checkpoint"
    )
    
    # 自定义策略网络
    policy_kwargs = {
        "features_extractor_class": CustomPolicy,
        "features_extractor_kwargs": {"features_dim": 512},  # 增大特征维度
        "net_arch": dict(
        pi=[512, 384, 256, 128],  # 策略网络
                    vf=[512, 384, 256, 128]),  # 价值函数网络
        "activation_fn": nn.ReLU,
        "ortho_init": True,  # 使用正交初始化
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
            n_steps=2048,
            batch_size=512,
            n_epochs=15,
            learning_rate=1e-4,
            clip_range=0.15,
            clip_range_vf=0.15,
            ent_coef=0.05,
            max_grad_norm=0.8,
            use_sde=True,  # 保留在这里
            sde_sample_freq=8,  # 添加SDE采样频率
            verbose=1,
            tensorboard_log="../logs/",
            #device="cuda" if torch.cuda.is_available() else "cpu"
            device="cpu"
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="../checkpoints/best/",
        log_path="../logs/",
        eval_freq=1000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        callback_on_new_best=StopTrainingOnRewardThreshold(reward_threshold=2500, verbose=1)
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
                callback=[checkpoint_callback, eval_callback],
                reset_num_timesteps=False,
                tb_log_name="PPO_exp1"
            )

            # 每20次迭代进行一次可视化评估
            #if i % 20 == 0:
            # 重置可视化环境
            obs = env.reset()[0]
            total_eval_reward = 0
            
            # 运行评估episode（最多350步）
            for eval_step in range(350):
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

        # 在for循环结束后添加最终评估
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
        model.policy.to("cpu")
        export_to_onnx(model, onnx_path, env.observation_space.shape)
        print("Training completed and model saved.")

    if viewer is not None:
        viewer.close()

if __name__ == "__main__":
    train()