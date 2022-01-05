import pyvirtualdisplay

_display = pyvirtualdisplay.Display(visible=False, size=(1400, 900)) #create an environment for the simulation to run in
_ = _display.start()

import gym
import ray
from ray import tune
#from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer  #switched training algorithm to PPO (Proximal Policy Optimisation
from ray.tune.registry import register_env

ray.shutdown()
ray.init(include_webui=False, ignore_reinit_error=True) #startup learning process

ENV = 'PoleBalance' #AIGym environment creation

def make_env(env_config):
    import pybullet_envs
    return gym.make(ENV)

register_env(ENV,make_env())
TARGET = 190
#Trainer = DQNTrainer algorithm switched to PPO
Trainer = PPOTrainer

tune.run(
    Trainer,
    stop={"Instance_Reward_AVG" : TARGET},
    config = {"env" : ENV, "workers" : 7, "gpus" : 1, "monitor" : True, "InstanceEvalNum" : 50}
)


