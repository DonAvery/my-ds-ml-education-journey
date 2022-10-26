import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

homework = bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5").to_runner()

svc = bentoml.Service("ml_homework", runners=[homework])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    result = homework.predict.run(input_series)
    return result