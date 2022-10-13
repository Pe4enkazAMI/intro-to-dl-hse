import numpy as np
from .base import Criterion
from .activations import LogSoftmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """
    def compute_output(self, input: np.array, target: np.array) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        batch_size = input.shape[0]
        feature_size = input.shape[1]
        factor = batch_size * feature_size
        inv_factor = 1/factor
        loss = (input - target)**2
        return inv_factor * loss.sum()

    def compute_grad_input(self, input: np.array, target: np.array) -> np.array:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        batch_size = input.shape[0]
        feature_size = input.shape[1]
        factor = batch_size * feature_size
        inv_factor = 1/factor
        return 2*inv_factor * (input - target)


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()

    def compute_output(self, input: np.array, target: np.array) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        probs = self.log_softmax.compute_output(input)
        class_mask = np.zeros(input.shape)
        class_mask[np.arange(target.shape[0]), target] = 1
        loss = -1/input.shape[0] * (class_mask * probs).sum()
        return loss

    def compute_grad_input(self, input: np.array, target: np.array) -> np.array:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        class_mask = np.zeros(input.shape)
        class_mask[np.arange(target.shape[0]), target] = 1
        return -1/input.shape[0] * self.log_softmax.compute_grad_input(input, class_mask)
