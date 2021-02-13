package anfis.network;

import anfis.ml.loss.LossFunction;
import anfis.ml.sampling.Sample;
import anfis.ml.sampling.Sampling;
import anfis.ml.stopping.StoppingCondition;

import java.util.List;

public class BatchANFIS extends ANFIS {

    private static final double DEFAULT_ETA_1 = 0.0001;
    private static final double DEFAULT_ETA_2 = 0.00005;

    public BatchANFIS(int numberOfRules) {
        super(numberOfRules, DEFAULT_ETA_1, DEFAULT_ETA_2);
    }

    public BatchANFIS(int numberOfRules, double eta1, double eta2, LossFunction lossFunction, StoppingCondition stoppingCondition) {
        super(numberOfRules, eta1, eta2, lossFunction, stoppingCondition);
    }

    @Override
    protected final List<List<Sample>> partition(List<Sample> samples) {
        return Sampling.partition(samples, samples.size());
    }
}
