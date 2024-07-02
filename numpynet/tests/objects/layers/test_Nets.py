
import numpy as np
from numpynet.objects import Linear
from numpynet.objects.functions.activation import sigmoid

class Net1():
    """Example 1 of fully-connected neural Network."""

    def __init__(self) -> None:
        self.l1 = Linear(3, 4)
        self.l2 = Linear(4, 4)
        self.l3 = Linear(4, 3)
        self.l4 = Linear(3, 1)

    def forward(self, x: np.ndarray):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x


def test_Net1():
    net = Net1()
    x = np.array([0.1, 0.2, 0.1]).reshape(3,1)
    y = net.forward(x)
    assert y.shape == (1,1)


class Net2():
    """Example 2 of fully-connected neural newtwork."""

    def __init__(self) -> None:
        self.l1 = Linear(2,2)
        self.l1.set_weight(np.array([[0.1, 0.2], [0.3, 0.4]]))
        self.l1.set_bias(np.array([0.25, 0.25]).reshape(2,1))
        self.l2 = Linear(2,2)
        self.l2.set_weight(np.array([[0.5, 0.7], [0.6, 0.8]]))
        self.l2.set_bias(np.array([0.35, 0.35]).reshape(2,1))

    def forward(self, x: np.ndarray):
        x = self.l1(x)
        x = self.l2(x)
        return x


def test_Net2():
    net = Net2()
    x = np.array([0.1, 0.5]).reshape(2,1)
    y = net.forward(x)
    print(y)
    assert y.shape == (2,1)


class Net3():
    """Example 3 of neural newtwork.
    
    Arquitecture of the model and weight from the "Roasting Coffee Example" in the
    Advanced Learning Algorithms of Andrew Ng.
    
    """

    def __init__(self) -> None:
        self.l1 = Linear(2,3)
        self.l1.set_weight(
            np.array(
                [[-8.94, -0.17],
                 [0.29, -7.34],
                 [12.89, 10.79]]
            )
        )
        self.l1.set_bias(np.array([-9.87, -9.28, 1.01]).reshape(3,1))
        self.l2 = Linear(3,1)
        self.l2.set_weight(
            np.array(
                [[-31.38, -27.86, -32.79]]
            )
        )
        self.l2.set_bias(np.array([15.54]).reshape(1,1))

    def forward(self, x: np.ndarray):
        x = self.l1(x)
        x = sigmoid(x)
        x = self.l2(x)
        x = sigmoid(x)
        return x


def test_Net3():
    net = Net3()
    x1 = np.array([-0.47, 0.42]).reshape(2,1)
    x2 = np.array([-0.47, 3.16]).reshape(2,1)
    y1 = net.forward(x1)
    y2 = net.forward(x2)
    assert y1.round(2) == 0.96
    assert (y2 * 1e8).round(2) == 3.03


if __name__ == "__main__":
    test_Net3()
