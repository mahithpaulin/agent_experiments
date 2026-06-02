import numpy as np
import pytest
from main import ExplainableAI
from utils import KernelRegression

def test_kernel_regression_fit_predict():
    X = np.array([[0], [1], [2]])
    y = np.array([0, 1, 2])
    kr = KernelRegression(bandwidth=1.0)
    kr.fit(X, y)
    preds = kr.predict(np.array([[1.5]]))
    assert preds.shape == (1,)
    assert preds[0] > 1 and preds[0] < 2

def test_kernel_regression_untrained_predict_raises():
    kr = KernelRegression()
    with pytest.raises(RuntimeError):
        kr.predict(np.array([[0]]))

def test_special_training_trait_bandwidth_decay():
    kr = KernelRegression(bandwidth=1.0)
    initial_bw = kr.bandwidth
    for epoch in range(5):
        from utils import special_training_trait
        special_training_trait(kr, epoch)
    assert kr.bandwidth <= initial_bw
    assert kr.bandwidth >= 0.05

def test_explainable_ai_fit_predict():
    X = np.linspace(-1, 1, 10).reshape(-1, 1)
    y = np.sin(3 * X).ravel()
    ai = ExplainableAI(bandwidth=0.5)
    ai.fit(X, y, epochs=3)
    preds = ai.predict(X)
    assert preds.shape == y.shape
    # Check predictions roughly correlate with y
    corr = np.corrcoef(preds, y)[0, 1]
    assert corr > 0.8

def test_explainable_ai_predict_before_fit_raises():
    ai = ExplainableAI()
    with pytest.raises(RuntimeError):
        ai.predict(np.array([[0]]))