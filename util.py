import numpy as np
from typing import List, Tuple
from Neuron import Neuron


def sigmoid(nums: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-1 * nums))


def sigmoidDeriv(nums: np.ndarray) -> np.ndarray:
    return sigmoid(nums) * (1 - sigmoid(nums))


def softmax(nums: np.ndarray) -> np.ndarray:
    if len(nums.shape) == 1:
        nums = nums[..., np.newaxis]
    max_subtracted = nums - np.max(nums, axis=0, keepdims=True) # prevent overflow 
    return np.exp(max_subtracted) / np.sum(np.exp(max_subtracted), axis=0, keepdims=True)


def softmaxDeriv(target: np.ndarray, output: np.ndarray) -> np.ndarray:
    o_shape = output.shape
    t_shape = target.shape
    if len(o_shape) == 1:
        output = output[..., np.newaxis]
    if len(t_shape) == 1:
        target = target[..., np.newaxis]
    return output - target


# Deprecated
def getWeightMatrix(layer: List[Neuron]) -> np.matrix:
    weights = [neuron.weights for neuron in layer]
    return np.stack(weights, axis=1)


def cross_entropy_loss(target: np.ndarray, output: np.ndarray) -> float:
    return np.sum(np.log(output) * target) * (-1 / len(target))


def convert_string_to_vector(data: str) -> List[int]:
    return [int(n) for n in data.split(",")]


def scale_vector(data: List[int]) -> List[int]:
    return [n / 255.0 * 0.99 + 0.01 for n in data]


def compute_output_vector(output: int) -> List[int]:
    return [0.99 if i == output else 0.01 for i in range(10)]

def compute_clean_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    dataset = open(path, "r")
    data = dataset.readlines()
    dataset.close()

    X = []
    Y = []

    for d in data:
        d = convert_string_to_vector(d)
        x, y = d[1: ], d[0]

        X.append(scale_vector(x))
        Y.append(compute_output_vector(y))

    return np.asarray(X), np.asarray(Y)

def randomize_order(X: np.ndarray, Y: np.ndarray):
    X_len = X.shape[1]
    combined_data_set = np.concatenate((X, Y), axis=1)
    np.random.shuffle(combined_data_set)
    rand_X, rand_Y = combined_data_set[:, :X_len], combined_data_set[:, X_len:]
    return rand_X, rand_Y