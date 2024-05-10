import numpy as np
from typing import List
from Neuron import Neuron


def sigmoid(nums: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-1 * nums))


def sigmoidDeriv(nums: np.ndarray) -> np.ndarray:
    return sigmoid(nums) * (1 - sigmoid(nums))


def softmax(nums: np.ndarray) -> np.ndarray:
    return np.exp(nums) / np.sum(np.exp(nums))


def softmaxDeriv(target: np.ndarray, output: np.ndarray) -> np.ndarray:
    return output - target


def getWeightMatrix(layer: List[Neuron]) -> np.matrix:
    weights = [neuron.weights for neuron in layer]
    return np.stack(weights, axis=1)


def cross_entropy_loss(target: np.ndarray, output: np.ndarray) -> float:
    return np.sum(np.log(output) * target) * (-1 / len(target))
