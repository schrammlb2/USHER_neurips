import numpy as np
import gym #type: ignore
import os, sys #type: ignore
from mpi4py import MPI #type: ignore
import pdb #type: ignore
from HER.arguments import get_args #type: ignore
# from HER.rl_modules.ddpg_agent import ddpg_agent
# from HER.rl_modules.model_normed_ddpg_agent import ddpg_agent
# from HER.rl_modules.sac_agent import ddpg_agent
# from HER.rl_modules.p2p_agent import ddpg_agent
from HER.rl_modules.sac_models import StateValueEstimator#type: ignore

import random#type: ignore
import torch#type: ignore

# from pomp.planners.plantogym import PlanningEnvGymWrapper, KinomaticGymWrapper#type: ignore

# from pomp.example_problems.doubleintegrator import doubleIntegratorTest#type: ignore
# from pomp.example_problems.pendulum import pendulumTest#type: ignore
# from pomp.example_problems.gym_pendulum_baseenv import PendulumGoalEnv#type: ignore

# # from pomp.example_problems.robotics.fetch.reach import FetchReachEnv#type: ignore
# from pomp.example_problems.robotics.fetch.push import FetchPushEnv#type: ignore
# # from pomp.example_problems.robotics.fetch.slide import FetchSlideEnv#type: ignore
# # from pomp.example_problems.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv#type: ignore


# from pomp.example_problems.robotics.hand.reach import HandReachEnv#type: ignore

# from gym_extensions.continuous.gym_navigation_2d.env_generator import Environment, EnvironmentCollection, Obstacle
from gym_extensions.continuous.gym_navigation_2d.env_generator import Environment#, EnvironmentCollection, Obstacle

from HER_mod.rl_modules.velocity_env import MultiGoalEnvironment, CarEnvironment#type: ignore
from HER_mod.rl_modules.car_env import RotationEnv, NewCarEnv, SimpleMovementEnvironment#type: ignore
from HER_mod.rl_modules.continuous_acrobot import ContinuousAcrobotEnv#type: ignore

import pickle#type: ignore

from gym.wrappers.time_limit import TimeLimit#type: ignore
from action_randomness_wrapper import ActionRandomnessWrapper, RepeatedActionWrapper#type: ignore
"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""

LOGGING = True


def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params

def launch(args):
    # create the ddpg_agent
    if args.env_name == "MultiGoalEnvironment":
        env = MultiGoalEnvironment("MultiGoalEnvironment", time=True, vel_goal=False)
    elif args.env_name == "MultiGoalEnvironmentVelGoal":
        env = MultiGoalEnvironment("MultiGoalEnvironment", time=True, vel_goal=True)
    elif "Car" in args.env_name:
        # env = CarEnvironment("CarEnvironment", time=True, vel_goal=False)
        env = TimeLimit(NewCarEnv(vel_goal=False), max_episode_steps=50)
        # env = TimeLimit(CarEnvironment("CarEnvironment", time=True, vel_goal=False), max_episode_steps=50)
    elif args.env_name == "Asteroids" :
        env = TimeLimit(RotationEnv(vel_goal=False), max_episode_steps=50)
    elif args.env_name == "AsteroidsVelGoal" :
        env = TimeLimit(RotationEnv(vel_goal=True), max_episode_steps=50)
    elif "SimpleMovement" in args.env_name:
        env = TimeLimit(SimpleMovementEnvironment(vel_goal=False), max_episode_steps=50)
    elif args.env_name == "PendulumGoal":
        env = TimeLimit(PendulumGoalEnv(g=9.8), max_episode_steps=200)
    # elif args.env_name == "FetchReach":
    #     env = TimeLimit(FetchReachEnv(), max_episode_steps=50)
    # elif args.env_name == "FetchPush":
    #     env = TimeLimit(FetchPushEnv(), max_episode_steps=50)
    # elif args.env_name == "FetchSlide":
    #     env = TimeLimit(FetchSlideEnv(), max_episode_steps=50)
    # elif args.env_name == "FetchPickAndPlace":
    #     env = TimeLimit(FetchPickAndPlaceEnv(), max_episode_steps=50)
    elif args.env_name == "HandReach":
        env = TimeLimit(HandReachEnv(), max_episode_steps=10)
    elif "Gridworld" in args.env_name: 
        # from continuous_gridworld import create_map_1#, random_blocky_map, two_door_environment, random_map
        # from alt_gridworld_implementation import create_test_map, random_blocky_map, two_door_environment, random_map #create_map_1,
        from solvable_gridworld_implementation import create_test_map, random_blocky_map, two_door_environment, random_map, create_map_1
        # from gridworld_reimplementation import random_map

        max_steps = 50 if "Alt" in args.env_name else 20
        if args.env_name == "TwoDoorGridworld":
            env=TimeLimit(two_door_environment(), max_episode_steps=50)
        else:
            if "RandomBlocky" in args.env_name:
                mapmaker = random_blocky_map
            elif "Random" in args.env_name:
                mapmaker = random_map
            elif "Test" in args.env_name: 
                mapmaker = create_test_map
            else: 
                mapmaker = create_map_1

            if "Asteroids" in args.env_name: 
                env_type="asteroids"
            elif "StandardCar" in args.env_name:
                env_type = "standard_car"
            elif "Car" in args.env_name:
                env_type = "car"
            else: 
                env_type = "linear"
            print(f"env type: {env_type}")
            env = TimeLimit(mapmaker(env_type=env_type), max_episode_steps=max_steps)
    elif "ContinuousAcrobot" in args.env_name:
        env = TimeLimit(ContinuousAcrobotEnv(), max_episode_steps=50)
    elif "2DNav" in args.env_name or "2Dnav" in args.env_name: 
        env = gym.make("Limited-Range-Based-Navigation-2d-Map8-Goal0-v0")
    else:
        env = gym.make(args.env_name)

    env = ActionRandomnessWrapper(env, args.action_noise)
    # env =  RepeatedActionWrapper(env, 5)
    # env = TimeLimit(FetchReachEnv(), max_episode_steps=50)
    # env = TimeLimit(FetchPushEnv(), max_episode_steps=50)
    # # problem = doubleIntegratorTest()
    # problem = pendulumTest()
    # # env = PlanningEnvGymWrapper(problem)
    # env = KinomaticGymWrapper(problem)
    # env = gym.make(args.env_name)
    # set random seeds for reproduce

    try: 
        env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    except: 
        pass
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # get the environment parameters
    env_params = get_env_params(env)
    # create the ddpg agent to interact with the environment 

    #inject randomness into environment
    # pdb.set_trace()
    # step_method = env.step
    # env.step = lambda action: step_method(
    #         action + np.random.normal(
    #             loc=[0]*env_params['action'], 
    #             scale=[args.action_noise]*env_params['action']
    #             )
    #         )

    ddpg_trainer = ddpg_agent(args, env, env_params)
    ddpg_trainer.learn()
    return ddpg_trainer

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    # print(args)
    # exit()

    if args.p2p: 
        # if "Fetch" in args.env_name:
        #     from HER.rl_modules.fetch_specific_p2p import ddpg_agent
        # else: 
        #     from HER.rl_modules.p2p_agent import ddpg_agent
        from HER.rl_modules.p2p_agent import ddpg_agent
        # from HER.rl_modules.tdm_p2p import ddpg_agent
        suffix = "_p2p"
    else: 
        # if args.two_goal:
        #     from HER.rl_modules.usher_agent import ddpg_agent
        # else:
        #     from HER.rl_modules.sac_agent import ddpg_agent
        # if args.apply_ratio: 
        #     from HER.rl_modules.generalized_usher_with_ratio_2 import ddpg_agent
        #     # from HER.rl_modules.generalized_usher_with_ratio import ddpg_agent
        # else:
        #     from HER.rl_modules.generalized_usher import ddpg_agent
        # if args.apply_ratio:
        #     from HER.rl_modules.theoretically_sound_agent import ddpg_agent
        # else: 
        #     from HER.rl_modules.generalized_usher_with_ratio_2 import ddpg_agent
        from HER.rl_modules.theoretically_sound_agent import ddpg_agent
        # from HER.rl_modules.joint_sampling_agent import ddpg_agent
        # from HER.rl_modules.mod_agent import ddpg_agent
        # from HER.rl_modules.heuristic_difference_sac_agent import ddpg_agent
        # from HER.rl_modules.value_prior_agent import ddpg_agent
        # from HER.rl_modules.ddpg_agent import ddpg_agent
        # print("Using the model from the transferred section")
        # from HER_RFF_SF.rl_modules.ddpg_original import ddpg_agent
        # from HER.rl_modules.tdm_agent import ddpg_agent
        # from HER.rl_modules.tdm_ddpg_agent import ddpg_agent
        suffix = ""

    agent = launch(args)

    # with open("saved_models/her_" + args.env_name + suffix + ".pkl", 'wb') as f:
    #     pickle.dump(agent.actor_network, f)
    #     print("Saved agent")

    # value_estimator = StateValueEstimator(agent.actor_network, agent.critic_network, args.gamma)

    # with open("saved_models/her_" + args.env_name + "_value" + suffix + ".pkl", 'wb') as f:
    #     pickle.dump(value_estimator, f)
    #     print("Saved value estimator")


    
    n = 10
    # success_rate, reward, value = ev['success_rate'], ev['reward_rate'], ev['value_rate']
    # success_rate = sum([agent._eval_agent(final=True)['success_rate'] for _ in range(n)])/n
    # if LOGGING and MPI.COMM_WORLD.Get_rank() == 0:
    #     # pdb.set_trace()
    #     log_file_name = f"logging/{args.env_name}.txt"
    #     # success_rate = sum([agent._eval_agent()[0] for _ in range(n)])/n
    #     text = f"action_noise: {args.action_noise}, \ttwo_goal: {args.two_goal}, \tsuccess_rate: {success_rate}\n"
    #     with open(log_file_name, "a") as f:
    #         f.write(text)

    #     print("Log written")
