package anfis.network;

import anfis.ml.loss.LossFunction;
import anfis.ml.sampling.Sample;
import anfis.ml.sampling.Sampling;
import anfis.ml.stopping.StoppingCondition;
import anfis.ml.sampling.SourceOfRandomness;

import java.util.Collections;
import java.util.List;

public class StochasticANFIS extends ANFIS {

    private static final double DEFAULT_ETA_1 = 0.001;
    private static final double DEFAULT_ETA_2 = 0.0005;

    private static final int DEFAULT_BATCH_SIZE = 1;

    private final int batchSize;

    public StochasticANFIS(int numberOfRules) {
        this(numberOfRules, DEFAULT_BATCH_SIZE);
    }

    public StochasticANFIS(int numberOfRules, int batchSize) {
        super(numberOfRules, DEFAULT_ETA_1, DEFAULT_ETA_2);
        this.batchSize = batchSize;
    }

    public StochasticANFIS(int numberOfRules, double eta1, double eta2, LossFunction lossFunction, StoppingCondition stoppingCondition, int batchSize) {
        super(numberOfRules, eta1, eta2, lossFunction, stoppingCondition);
        this.batchSize = batchSize;
    }

    @Override
    protected final void preprocess(List<Sample> samples) {
        Collections.shuffle(samples, SourceOfRandomness.getSource());
    }

    @Override
    protected final List<List<Sample>> partition(List<Sample> samples) {
        return (batchSize == 1) ? Sampling.singletons(samples) : Sampling.partition(samples, batchSize);
    }
}
