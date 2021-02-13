package anfis.ml.exceptions;

import anfis.ml.MachineLearningModel;

public class ModelNotFittedException extends RuntimeException {

    public ModelNotFittedException(Class<? extends MachineLearningModel> modelClass) {
        super("This instance of " + modelClass.getSimpleName() + " has not been fitted yet");
    }
}
