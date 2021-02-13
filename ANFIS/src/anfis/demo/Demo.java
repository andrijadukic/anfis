package anfis.demo;

import anfis.network.StochasticANFIS;
import anfis.ml.loss.LossFunction;
import anfis.ml.loss.LossFunctions;
import anfis.ml.sampling.Sample;
import anfis.ml.stopping.StoppingCondition;
import anfis.ml.stopping.StoppingConditions;
import anfis.ml.observers.NthIterationObserver;
import anfis.ml.observers.StandardOutputLogger;

import java.io.IOException;
import java.util.*;

public class Demo {

    public interface MultivariateFunction {

        double valueAt(double... input);
    }

    public static final int LOWER_BOUND = -4;
    public static final int UPPER_BOUND = 4;

    public static final MultivariateFunction FUNCTION = input -> {
        double x = input[0];
        double y = input[1];
        return (Math.pow(x - 1, 2) + Math.pow(y + 2, 2) - 5 * x * y + 3) * Math.pow(Math.cos(x / 5), 2);
    };

    private static final int NUMBER_OF_RULES = 9;
    private static final double ETA_1 = 0.001;
    private static final double ETA_2 = 0.0005;
    private static final LossFunction LOSS_FUNCTION = LossFunctions.MSE();
    private static final StoppingCondition MAX_ITER_LARGE = StoppingConditions.maxIter(50_000);

    public static void main(String[] args) throws IOException {
        List<Sample> trainingSamples = sample(FUNCTION, LOWER_BOUND, UPPER_BOUND);

        var stochasticANFIS = new StochasticANFIS(NUMBER_OF_RULES, ETA_1, ETA_2, LOSS_FUNCTION, MAX_ITER_LARGE, 1);

        stochasticANFIS.addObserver(new NthIterationObserver(new StandardOutputLogger(), 5000));

        stochasticANFIS.fit(trainingSamples);
    }

    public static List<Sample> sample(MultivariateFunction function, int lb, int ub) {
        List<Sample> samples = new ArrayList<>();
        for (double x = lb; x <= ub; x++) {
            for (double y = lb; y <= ub; y++) {
                samples.add(new Sample(new double[]{x, y}, function.valueAt(x, y)));
            }
        }
        return samples;
    }
}
