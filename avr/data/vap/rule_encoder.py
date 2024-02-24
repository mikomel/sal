from abc import ABC, abstractmethod

import numpy as np
import torch


def create_vap_rule_encoder(name: str) -> "VAPRuleEncoder":
    if name == "dense":
        return DenseVAPRuleEncoder()
    elif name == "sparse":
        return SparseVAPRuleEncoder()
    else:
        raise ValueError(
            f"Can't create VAPRuleEncoder with name {name}. Choose one from: {{dense, sparse}}"
        )


class VAPRuleEncoder(ABC):
    @staticmethod
    @abstractmethod
    def encode(data: np.array) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def encoding_size() -> int:
        pass


class DenseVAPRuleEncoder(VAPRuleEncoder):
    @staticmethod
    def encode(data: np.array) -> torch.Tensor:
        return torch.from_numpy(np.squeeze(data["relation_structure_encoded"])).float()

    @staticmethod
    def encoding_size() -> int:
        return 12


class SparseVAPRuleEncoder(VAPRuleEncoder):
    @staticmethod
    def encode(data: np.array) -> torch.Tensor:
        """
        Format of relation_structure_encoded:
        [shape, line, color, number, position, size, type, progression, XOR, OR, AND, _not used_]
        """
        structure = data["relation_structure_encoded"]
        rules = torch.zeros(SparseVAPRuleEncoder.encoding_size()).float()
        indices = structure[0, :].nonzero()[0]
        if len(indices) == 3:
            if indices[0] == 1 and indices[1] == 2:  # line color
                idx = indices[2] - 7
            elif indices[0] == 1 and indices[1] == 6:  # line type
                idx = 4 + indices[2] - 7
            else:  # shape
                idx = 8 + (indices[1] - 2) * 4 + (indices[2] - 7)
            rules[idx] = 1.0
        return rules

    @staticmethod
    def encoding_size() -> int:
        return 28
