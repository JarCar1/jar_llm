from agents.common import Agent, Actor
from utils.pytorch_util_SQL import from_numpy
from torch import Tensor

import torch
import numpy as np

class PGAgent(Agent):
    def __init__(
        self,
        actor: Actor,
        gamma: float,
        learning_rate: float,
        normalize_advantages: bool,
    ):
        super().__init__()

        # init actor (policy)
        self.actor = actor
        self.lr = learning_rate

        # other agent parameters
        self.EPSILON = 1e-4
        self.gamma = gamma
        self.normalize_advantages = normalize_advantages

    def get_action(self, observations):
        return self.actor.get_action(observations)

    def update(
        self,
        observations: Tensor,
        actions: Tensor,
        next_observations: Tensor,
        rewards: Tensor,
        dones: Tensor,
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Input is batch of (ob, ac, n_ob, rew, done) groups, where batch items are ordered by occurrence in a trajectory.
        Batches must end in a terminal (i.e. dones[-1] == 1).
        """
        batch_size = dones.shape[0]
        assert len(dones.shape) == 1, f"Bad batch shape: {dones.shape}"
        assert dones[-1] == 1, f"Batch does not end in a terminal state"

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        # with a Monte Carlo estimate
        terminal_indices: Tensor = dones.nonzero(as_tuple=True)[0]
        traj_lens = terminal_indices.diff(prepend=torch.tensor([-1]))
        longest_len = traj_lens.max().item()

        T = longest_len
        t, t_prime = np.mgrid[0:T, 0:T]
        diffs = (t_prime - t)
        discount_matrix = from_numpy((self.gamma ** diffs) * (diffs >= 0))

        traj_starts = torch.concat((torch.tensor([0]), terminal_indices[:-1] + 1))
        traj_ends = terminal_indices + 1

        q_values = torch.concat([
            (discount_matrix[:traj_len, :traj_len] * rewards[traj_start:traj_end]).sum(axis=1)
            for traj_start, traj_end, traj_len in zip(traj_starts, traj_ends, traj_lens)
        ])

        assert q_values.shape == rewards.shape, f"Q value calculation failed!: q shape {q_values.shape} ;; rew shape {rewards.shape}"

        # step 2: calculate advantages from Q values
        advantages = q_values
        if self.normalize_advantages:
            demeaned_adv = advantages - advantages.mean()
            adv_std_and_eps = advantages.std() + self.EPSILON

            advantages = demeaned_adv / adv_std_and_eps

        # step 3: update the PG actor/policy
        info: dict = self.actor.update(observations, actions, next_observations, advantages, dones) 

        return info
