package ml.loss;

import ml.Predictor;
import ml.sampling.Sample;

import java.util.List;

@FunctionalInterface
public interface LossFunction {

    double score(Predictor model, List<Sample> samples);
}
