package anfis.ml.loss;

import anfis.ml.Predictor;
import anfis.ml.sampling.Sample;

import java.util.List;

@FunctionalInterface
public interface LossFunction {

    double score(Predictor model, List<Sample> samples);
}
