import os
import time

from agents.pg_agent import PGAgent
from envs.number_station import NSSQLEnv
from envs.spider import SpiderSQLEnv

from envs.number_station import NSDataset
from envs.spider import SpiderDataset

from torch.utils.data import DataLoader


import os
import time

import numpy as np
import torch


from utils import pytorch_util_SQL as ptu
from utils.logger import Logger
from utils import tokenizer
from utils import util_SQL
from networks.policies import MLPPolicyPG


from utils.tokenizer import CodeT5p2BTokenizer  
from utils.tokenizer import CodeT5p770MTokenizer  
from utils.tokenizer import CodeT5p220MTokenizer  


def run_training_loop(args):
    logger = Logger(args.logdir)

    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)
    
    
    dataset = NSDataset('NSS_file.pkl')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # Adjust batch_size as needed
    
    # Create an instance of the tokenizer
    tokenizer = CodeT5p2BTokenizer()

    # make the gym environment
    env = NSSQLEnv(dataloader, tokenizer)
    
    #env = NSSQLEnv(self, tokenizer)

    #max_ep_len = args.ep_len or env.max_length
    
    max_ep_len = env.max_length
    
    
    ac_dim = 
    ob_dim = 
    discrete = 
    n_layers = 
    layer_size = 
    learning_rate = 
    
    actor = MLPPolicyPG(ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate)
    pg_agent = PGAgent(actor, gamma, learning_rate, normalize_advantages)

    # initialize agent
    agent = PGAgent(
        actor=Actor,
        gamma=args.discount,
        learning_rate=args.learning_rate,
        normalize_advantages=args.normalize_advantages
    )

    total_envsteps = 0
    start_time = time.time()

    for itr in range(args.n_iter):
        print(f"\n********** Iteration {itr} ************")
        # sample `args.batch_size` transitions using util_SQL.sample_trajectories
        # make sure to use `max_ep_len`
        trajs, envsteps_this_batch = None, None 
        total_envsteps += envsteps_this_batch

        # trajs should be a list of dictionaries of NumPy arrays, where each dictionary corresponds to a trajectory.
        # this line converts this into a single dictionary of lists of NumPy arrays.
        trajs_dict = {k: [traj[k] for traj in trajs] for k in trajs[0]}

        # train the agent using the sampled trajectories and the agent's update function
        train_info: dict = None

        if itr % args.scalar_log_freq == 0:
            # save eval metrics
            print("\nCollecting data for eval...")
            eval_trajs, eval_envsteps_this_batch = util_SQL.sample_trajectories(
                env, agent.actor, args.eval_batch_size, max_ep_len
            )

            logs = util_SQL.compute_metrics(trajs, eval_trajs)
            # compute additional metrics
            logs.update(train_info)
            logs["train_env_steps"] = total_envsteps
            logs["time_elapsed"] = time.time() - start_time
            if itr == 0:
                logs["Initial_DataCollection_AverageReturn"] = logs[
                    "Train_AverageReturn"
                ]

            # perform the logging
            for key, value in logs.items():
                print("{} : {}".format(key, value))
                logger.log_scalar(value, key, itr)
            print("Done logging...\n\n")

            logger.flush()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default = "english_to_SQL") #required=True)
    parser.add_argument("--n_iter", "-n", type=int, default=200)

    parser.add_argument("--normalize_advantages", "-na", action="store_true")
    parser.add_argument(
        "--batch_size", "-b", type=int, default=1000
    )  # steps collected per train iteration
    parser.add_argument(
        "--eval_batch_size", "-eb", type=int, default=400
    )  # steps collected per eval iteration

    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-3)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--scalar_log_freq", type=int, default=1)

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "exp_"
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/")
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = (
        logdir_prefix
        + args.exp_name
        + "_"
        + time.strftime("%d-%m-%Y_%H-%M-%S")
    )
    logdir = os.path.join(data_path, logdir)
    args.logdir = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)


    run_training_loop(args)


if __name__ == "__main__":
    main()
