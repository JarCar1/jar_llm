from torch import nn, Tensor

class Agent(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def get_action(self, observations: Tensor) -> Tensor:
        raise NotImplementedError("`Agent` is an abstract class and cannot `get_action`")

    def update(self, observations: Tensor, actions: Tensor, next_observations: Tensor, rewards: Tensor, dones: Tensor) -> dict:
        raise NotImplementedError("`Agent` is an abstract class and cannot `update`")


class Actor(nn.Module):
    """
    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("`Actor` is an abstract class and cannot perform a `forward` pass!")
    
    def get_action(self, observations: Tensor) -> Tensor:
        raise NotImplementedError("`Actor` is an abstract class and cannot `get_action`")
    
    def update(self, observations: Tensor, actions: Tensor, next_observations: Tensor, rewards: Tensor, dones: Tensor) -> dict:
        raise NotImplementedError("`Actor` is an abstract class and cannot be `update`-ed!")

