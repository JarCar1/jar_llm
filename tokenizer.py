
import abc
from torch import Tensor
from typing import List
from transformers import AutoTokenizer

class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def to_tokens(self, strings: List[str]) -> Tensor:
        raise NotImplementedError("to_tokens is an abstract method!")
    
    @abc.abstractmethod
    def from_tokens(self, tokens: Tensor) -> List[str]:
        raise NotImplementedError("from_tokens is an abstract method!")
    
    def render_string(self, tokens: Tensor) -> List[str]:
        return self.from_tokens(tokens)
    
    def render_tokens(self, strings: List[str]) -> Tensor:
        return self.to_tokens(strings)

class ASCIITokenizer(Tokenizer):

    def to_tokens(self, strings: List[str]) -> Tensor:
        return Tensor(
            list(map(
                lambda string: list(map(ord, string)),
                strings
            ))
        )

    def from_tokens(self, tokens: Tensor) -> List[str]:
        return [    
            "".join([chr(int(code)) for code in token_seq])
            for token_seq in tokens 
        ]

class HuggingFaceTokenizer(Tokenizer):
    def __init__(self, checkpoint_name):
        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_name, padding_side="left"
        )
    
    def to_tokens(self, strings: List[str]) -> Tensor:
        return self.hf_tokenizer(strings, return_tensors="pt", padding=True)["input_ids"]

    def from_tokens(self, tokens: Tensor) -> List[str]:
        return self.hf_tokenizer.batch_decode(tokens, skip_special_tokens=True)

class CodeT5p2BTokenizer(HuggingFaceTokenizer):
    def __init__(self) -> None:
        super(CodeT5p2BTokenizer, self).__init__("Salesforce/codet5p-2b")

class CodeT5p770MTokenizer(HuggingFaceTokenizer):
    def __init__(self) -> None:
        super(CodeT5p770MTokenizer, self).__init__("Salesforce/codet5p-770m")

class CodeT5p220MTokenizer(HuggingFaceTokenizer):
    def __init__(self) -> None:
        super(CodeT5p220MTokenizer, self).__init__("Salesforce/codet5p-220m")
