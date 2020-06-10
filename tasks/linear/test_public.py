import numpy as np
import pytest

from .cases import Case, cases
from .layer import Linear

# In case if tf and torch will be in environment
# @pytest.fixture(autouse=True)
# def test_numpy_only(monkeypatch) -> None:
#     monkeypatch.delattr('torch.nn.Linear')
#     monkeypatch.delattr('tensorflow.nn.layers.Linear')
#     monkeypatch.delattr('tensorflow.keras.layers.Dense')


@pytest.mark.timeout(0.01)
@pytest.mark.parametrize('t', cases, ids=str)
def test_torch_equality(t: Case) -> None:
    n_in, n_out = 3, 4
    custom_layer = Linear(n_in, n_out)
    custom_layer.W = t.weight
    custom_layer.b = t.bias

    custom_layer_output = custom_layer.updateOutput(t.input)
    assert np.allclose(t.torch_layer_output, custom_layer_output, atol=1e-6)

    custom_layer_grad = custom_layer.updateGradInput(t.input, t.next_layer_grad)
    assert np.allclose(t.torch_layer_grad, custom_layer_grad, atol=1e-6)

    custom_layer.accGradParameters(t.input, t.next_layer_grad)
    assert np.allclose(t.torch_weight_grad, custom_layer.gradW, atol=1e-6)
    assert np.allclose(t.torch_bias_grad, custom_layer.gradb, atol=1e-6)
