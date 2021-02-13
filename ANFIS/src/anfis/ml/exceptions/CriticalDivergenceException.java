package anfis.ml.exceptions;

import anfis.ml.MachineLearningModel;

public class CriticalDivergenceException extends RuntimeException {

    public CriticalDivergenceException(Class<? extends MachineLearningModel> modelClass) {
        super("Critical divergence occurred while fitting this instance of " + modelClass.getSimpleName());
    }
}
