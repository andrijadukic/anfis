package anfis.ml;


import anfis.ml.sampling.Sample;

import java.util.List;

public interface MachineLearningModel extends Predictor {

    MachineLearningModel fit(List<Sample> samples);
}
