import sqlite3
from torch.utils.data import DataLoader
from torch import Tensor
from utils.tokenizer import Tokenizer
from typing import List, Tuple, Dict
import torch

def sql_script_is_executable(script: str) -> bool:
    with sqlite3.connect(":memory:") as conn:
        try:
            conn.executescript(script)
        except (sqlite3.OperationalError, sqlite3.IntegrityError) as e:
            return False
    return True


class SQLEnv:
    """
    Abstract class SQL query builder that acts as a gym 
    
    Inheriting classes must implement `_calculate_reward()`
    """
    END_TOKEN = ";"

    def __init__(self, dataloader: DataLoader, tokenizer: Tokenizer, context_length=10_000, last_token_index: int=-1):
        self.data = iter(dataloader)
        self.tokenizer = tokenizer
        self.batch_size = dataloader.batch_size

        # we pick the `last_token_index because tokenizers sometimes add an eos token
        self.END_TOKEN = tokenizer.to_tokens(
                [SQLEnv.END_TOKEN] * self.batch_size
            )[:, last_token_index].view((self.batch_size, ))
        self.last_token_index = last_token_index

        self.max_length = context_length # count of tokens in the whole sql script
        
    def reset(self) -> Tensor:
        prompts, solutions = next(self.data)
        
        self.states = self.tokenizer.to_tokens(prompts)
        if self.last_token_index < -1:
            self.states = self.states[:, :self.last_token_index+1]
        self.labels = torch.cat(
            (self.states,
            self.tokenizer.to_tokens(solutions),),
            dim=1 # along the time axis
        )

        return self.states

    def step(self, ac_tokens: Tensor) -> Tuple[Tensor, Tensor, Tensor, Dict]:
        assert (
            ac_tokens.shape[0] == self.states.shape[0]
        ), f"Action batch size {ac_tokens.shape[0]} does not match internal batch size {self.states.shape[0]}"

        self.states = torch.cat(
            (self.states,
            ac_tokens),
            dim=1 # along the time axis
        )

        dones = torch.logical_or(
            torch.eq(self.states[:, -1].view((self.batch_size, )), self.END_TOKEN),
            torch.full((self.batch_size,), self.states.shape[-1] >= self.max_length)
        )

        assert dones.shape == (self.batch_size, ), f"Wrong dones: {dones.shape} -- {dones}"

        rewards = self._calculate_rewards()

        return self.states, rewards, dones, {}

    def _calculate_rewards(self) -> Tensor:
        raise NotImplementedError("Reward must be calculated for each dataset differently!")

    def render(self):
        for string in self.tokenizer.from_tokens(self.states):
            print(f"{string}")
