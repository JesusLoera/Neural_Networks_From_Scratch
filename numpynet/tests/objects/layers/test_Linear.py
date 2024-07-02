
import pytest
import numpy as np
from numpynet.objects import Linear


def test_valid_instance():
    l1 = Linear(in_features=3, out_features=5)
    assert l1.in_features == 3
    assert l1.out_features == 5
    assert l1.weight.shape == (5, 3)
    assert l1.bias.shape == (5, 1)


def test_weight_setter():
    l1 = Linear(in_features=5, out_features=4)
    new_weight_matrix = np.full((4,5), 0.1)
    l1.set_weight(new_weight_matrix)
    rows, columns = l1.weight.shape
    for i in range(rows):
        for j in range(columns):
            assert l1.weight[i][j] == 0.1


def test_invalid_input_vectors():
    l1 = Linear(in_features=4, out_features=3)
    with pytest.raises(ValueError):
        x = np.random.rand(3, 1)
        l1(x)
    with pytest.raises(ValueError):
        x = np.random.rand(5, 1)
        l1(x)


def test_output_vector_computation():
    l1 = Linear(in_features=4, out_features=3)
    custom_weight = np.array(
        [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12]]
    )
    l1.set_weight(custom_weight)
    custom_bias = np.array([0.2, 0.4, 0.6,]).reshape((3,1))
    l1.set_bias(custom_bias)
    x = np.array([0.1, 0.2, 0.3, 0.4]).reshape(4,1)
    y = l1(x)
    assert y[0][0] == 3.2
    assert y[1][0] == 7.4
    assert y[2][0] == 11.6
