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
# 在文件顶部导入模块处添加
import sys  # 新增
from datetime import datetime  # 如果尚未导入

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
    class WrappedPolicy(nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, x):
            return self.policy(x, deterministic=True)  # 固定deterministic参数

    device = torch.device("cpu")
    model.policy = model.policy.to(device).eval()
    
    # 包装策略
    wrapped_policy = WrappedPolicy(model.policy)
    
    with torch.no_grad():
        for param in wrapped_policy.parameters():
            param.requires_grad_(False)
        
        dummy_input = torch.randn(1, *input_shape, dtype=torch.float32).to(device)
        
        if hasattr(model.policy, 'reset_noise'):
            model.policy.reset_noise()
            model.policy.sde_mode = False
        
        torch.onnx.export(
            wrapped_policy,  # 使用包装后的策略
            dummy_input,
            output_path,
            opset_version=12,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL
        )
        
def train():
    parser = argparse.ArgumentParser()
        # +++ 日志初始化代码 +++
    
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"../logs/training_{start_time}.log"
    
    class TeeLogger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a", buffering=1)  # 行缓冲模式
            
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            
        def flush(self):
            self.terminal.flush()
            self.log.flush()
            
        def close(self):
            self.log.close()
            
    original_stdout = sys.stdout
    tee = TeeLogger(log_file)
    sys.stdout = tee

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
        callback_on_new_best=StopTrainingOnRewardThreshold(reward_threshold=10000, verbose=1)
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
            # 重置可视化环境
            reset_result = env.reset()  # 返回 (obs, info)
            obs = reset_result[0]  # 提取 obs
            total_eval_reward = 0

            # 运行评估 episode（最多350步）
            for eval_step in range(350):
                action, _ = model.predict(obs, deterministic=False)  # 使用提取出的 obs
                step_result = env.step(action)  # 返回 (obs, reward, terminated, truncated, info)
                obs = step_result[0]  # 提取新的 obs
                reward = step_result[1]
                terminated = step_result[2]
                truncated = step_result[3]
                total_eval_reward += reward  # 直接累加这个 reward 值

                # 更新可视化
                if hasattr(viewer, 'is_running') and viewer.is_running():  # 确保 viewer 未关闭
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
        # 在for循环结束后添加最终评估
        if 4999 % 100 != 0:  # 如果最后一次迭代不是评估周期
            # 执行最终评估
            reset_result = env.reset()  # 返回 (obs, info)
            obs = reset_result[0]  # 提取 obs
            total_eval_reward = 0
            for eval_step in range(200):
                action, _ = model.predict(obs)
                step_result = env.step(action)
                obs = step_result[0]  # 提取 obs
                reward = step_result[1]
                terminated = step_result[2]
                truncated = step_result[3]
                total_eval_reward += reward
                if hasattr(viewer, 'is_running') and viewer.is_running():
                    viewer.sync()
                if terminated or truncated:
                    break
            print(f"Final evaluation - Total reward: {total_eval_reward:.2f}")
                
# 在finally块中添加额外处理
    finally:
        if viewer is not None:
            viewer.close()
        
        sys.stdout = original_stdout
        tee.close()
        
        # 保存并重新加载模型以确保状态干净
        model.save("../checkpoints/ppo_final.zip")
        
        # 重新加载模型并显式设置参数
        model = PPO.load(
            "../checkpoints/ppo_final.zip",
            device="cpu",
            # 添加自定义对象处理
            custom_objects={
                'policy_kwargs': policy_kwargs,
                'observation_space': env.observation_space,
                'action_space': env.action_space
            }
        )
        
        # 显式设置策略网络模式
        model.policy.eval()
        for param in model.policy.parameters():
            param.requires_grad_(False)
        
        # 执行导出
        export_to_onnx(model, onnx_path, env.observation_space.shape)

if __name__ == "__main__":
    train()