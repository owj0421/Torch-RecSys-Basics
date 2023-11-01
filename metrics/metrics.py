import numpy as np
import numpy.typing as npt
from typing import Tuple, List, Dict

EPSILON = 0.001

def _check_inputs(predictions, references):
    if not isinstance(predictions, np.ndarray) or not isinstance(references, np.ndarray):
        raise TypeError(
            "Inputs must Numpy Array!"
        )
    if len(predictions) != len(references):
        raise ValueError(
            "Unmatched Length for Predictions and References!"
        )
    return predictions, references

class MetricCalculator():
    @staticmethod
    def rmse(
        predictions: npt.NDArray, references: npt.NDArray, verbose=True
        ) -> Dict[str, float]:
        """RMSE (Root Mean Squared Error)"""
        predictions, references = _check_inputs(predictions, references)
        mse_ = np.average((predictions - references) ** 2)
        rmse_ = np.sqrt(mse_)
        if verbose:
            print(f"RMSE : {rmse_:1.4f}")
        return {"rmse": rmse_}
    
    @staticmethod
    def mae(
        predictions: npt.NDArray, references: npt.NDArray, verbose=True
        ) -> Dict[str, float]:
        """MAE (Mean Absolute Error)"""
        predictions, references = _check_inputs(predictions, references)
        mae_ = np.average(np.abs(predictions - references))
        if verbose:
            print(f"MAE : {mae_:1.4f}")
        return {"mae": mae_}
    
if __name__ == '__main__':
    predictions = np.array([1, 2, 3, 4])
    references = np.array([10, 6, 7, 8])
    MetricCalculator().rmse(predictions, references)
    MetricCalculator().mae(predictions, references)