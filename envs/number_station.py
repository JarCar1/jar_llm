from torch.utils.data import Dataset
from torch import Tensor, tensor
import pandas as pd
from typing import Tuple
from envs.common import SQLEnv, sql_script_is_executable

class NSDataset(Dataset):
    """ A torch.Dataset implementation of the Number Station Text2SQL dataset"""
    def __init__(self, pkl_file: str):
        """Constructs an instance from a pickle of a pandas.DataFrame that 
           contains the columns 'instruction', 'output', and 'source'"""
        self.data: pd.DataFrame = pd.read_pickle(pkl_file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> Tuple[str, str]:
        point: pd.Series = self.data.iloc[index]
        # point = point[["instruction", "output", "source"]]
        return point['instruction'], point['output']

class NSSQLEnv(SQLEnv):
    """An SQL query builder that acts as a gym"""
    def _calculate_rewards(self) -> Tensor:
        # improve the reward function
        return tensor([
            sql_script_is_executable(script)
            for script in self.tokenizer.render_string(self.states)
        ], dtype=float)
