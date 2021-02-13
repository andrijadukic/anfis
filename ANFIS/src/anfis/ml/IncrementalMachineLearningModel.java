package anfis.ml;

import anfis.ml.sampling.Sample;

import java.util.List;

public interface IncrementalMachineLearningModel extends MachineLearningModel {

    IncrementalMachineLearningModel partialFit(List<Sample> samples);
}
