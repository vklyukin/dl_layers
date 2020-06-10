import numpy as np
import pytest

from .cases import Case, cases
from .layer import ReLU

# In case if tf and torch will be in environment
# @pytest.fixture(autouse=True)
# def test_numpy_only(monkeypatch) -> None:
#     monkeypatch.delattr('torch.nn.ReLU')
#     monkeypatch.delattr('torch.nn.functional.relu')
#     monkeypatch.delattr('tensorflow.nn.relu')


@pytest.mark.timeout(0.5)
@pytest.mark.parametrize('t', cases, ids=str)
def test_torch_equality(t: Case) -> None:
    custom_layer = ReLU()

    custom_layer_output = custom_layer.updateOutput(t.input)
    assert np.allclose(t.torch_layer_output, custom_layer_output, atol=1e-6)

    custom_layer_grad = custom_layer.updateGradInput(t.input, t.next_layer_grad)
    assert np.allclose(t.torch_layer_grad, custom_layer_grad, atol=1e-6)
