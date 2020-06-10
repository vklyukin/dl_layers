import numpy as np
import pytest

from .layer import Dropout

# In case if tf and torch will be in environment
# @pytest.fixture(autouse=True)
# def test_numpy_only(monkeypatch) -> None:
#     monkeypatch.delattr('torch.nn.Dropout')
#     monkeypatch.delattr('torch.nn.functional.dropout')
#     monkeypatch.delattr('tensorflow.keras.layers.Dropout')
#     monkeypatch.delattr('tensorflow.compat.v1.Dropout')
#     monkeypatch.delattr('tensorflow.nn.Dropout')


@pytest.mark.timeout(0.1)
def test_train() -> None:
    batch_size, n_in = 2, 4
    for _ in range(50):
        p = np.random.uniform(0.3, 0.7)
        layer = Dropout(p)
        layer.train()

        layer_input = np.random.uniform(-5, 5, (batch_size, n_in)).astype(np.float32)
        next_layer_grad = np.random.uniform(-5, 5, (batch_size, n_in)).astype(np.float32)

        # check layer output
        layer_output = layer.updateOutput(layer_input)
        assert np.all(np.logical_or(np.isclose(layer_output, 0),
                                    np.isclose(layer_output * (1. - p), layer_input)))

        # check layer input gradient
        layer_grad = layer.updateGradInput(layer_input, next_layer_grad)
        assert np.all(np.logical_or(np.isclose(layer_grad, 0),
                                    np.isclose(layer_grad * (1. - p), next_layer_grad)))


@pytest.mark.timeout(0.01)
def test_eval() -> None:
    batch_size, n_in = 2, 4
    for _ in range(10):
        p = np.random.uniform(0.3, 0.7)
        layer = Dropout(p)

        layer_input = np.random.uniform(-5, 5, (batch_size, n_in)).astype(np.float32)
        layer.evaluate()
        layer_output = layer.updateOutput(layer_input)
        assert np.allclose(layer_output, layer_input)


@pytest.mark.timeout(0.1)
def test_mask() -> None:
    batch_size, n_in = 2, 4
    for _ in range(50):
        layer_input = np.random.uniform(-5, 5, (batch_size, n_in)).astype(np.float32)

        p = 0.0
        layer = Dropout(p)
        layer.train()
        layer_output = layer.updateOutput(layer_input)
        assert np.allclose(layer_output, layer_input)

        p = 0.5
        layer = Dropout(p)
        layer.train()
        layer_input = np.random.uniform(5, 10, (batch_size, n_in)).astype(np.float32)
        next_layer_grad = np.random.uniform(5, 10, (batch_size, n_in)).astype(np.float32)
        layer_output = layer.updateOutput(layer_input)
        zeroed_elem_mask = np.isclose(layer_output, 0)
        layer_grad = layer.updateGradInput(layer_input, next_layer_grad)
        assert np.all(zeroed_elem_mask == np.isclose(layer_grad, 0))


@pytest.mark.timeout(0.01)
def test_iid_mask() -> None:
    for _ in range(10):
        batch_size, n_in = 1000, 1
        p = 0.8
        layer = Dropout(p)
        layer.train()

        layer_input = np.random.uniform(5, 10, (batch_size, n_in)).astype(np.float32)
        layer_output = layer.updateOutput(layer_input)
        assert np.sum(np.isclose(layer_output, 0)) != layer_input.size

        layer_input = layer_input.T
        layer_output = layer.updateOutput(layer_input)
        assert np.sum(np.isclose(layer_output, 0)) != layer_input.size
