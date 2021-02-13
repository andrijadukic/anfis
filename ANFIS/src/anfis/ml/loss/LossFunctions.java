package anfis.ml.loss;

import anfis.ml.exceptions.InvalidSampleException;
import anfis.ml.sampling.Sample;

import java.util.Arrays;

public final class LossFunctions {

    private LossFunctions() {
    }

    public static LossFunction MSE() {
        return (model, samples) -> {
            double mse = 0.;
            for (Sample sample : samples) {
                double[] target = sample.y();
                double[] output = model.predict(sample.x());

                if (target.length != output.length) throw new InvalidSampleException();

                for (int i = 0, n = output.length; i < n; i++) {
                    double diff = target[i] - output[i];
                    mse += diff * diff;
                }
            }
            return mse / samples.size();
        };
    }

    public static LossFunction zeroOneLoss() {
        return (model, samples) -> {
            int wrong = 0;
            for (Sample sample : samples) {
                double[] target = sample.y();
                double[] output = model.predict(sample.x());

                if (target.length != output.length) throw new InvalidSampleException();

                if (!Arrays.equals(sample.y(), model.predict(sample.x()))) {
                    wrong++;
                }
            }
            return (double) wrong / samples.size();
        };
    }
}
