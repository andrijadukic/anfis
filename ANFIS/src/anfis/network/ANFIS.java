package anfis.network;

import anfis.ml.IncrementalMachineLearningModel;
import anfis.ml.exceptions.*;
import anfis.ml.loss.LossFunction;
import anfis.ml.loss.LossFunctions;
import anfis.ml.observers.AbstractMachineLearningSubject;
import anfis.ml.observers.IterationStatistics;
import anfis.ml.sampling.Sample;
import anfis.ml.stopping.StoppingCondition;
import anfis.ml.stopping.StoppingConditions;
import anfis.ml.sampling.SourceOfRandomness;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public abstract class ANFIS extends AbstractMachineLearningSubject implements IncrementalMachineLearningModel {

    private static final LossFunction DEFAULT_LOSS_FUNCTION = LossFunctions.MSE();
    private static final StoppingCondition DEFAULT_STOPPING_CONDITION = StoppingConditions.maxIter(50_000);

    private final int numberOfRules;
    private final LossFunction lossFunction;
    private final StoppingCondition stoppingCondition;

    private int dimension;
    private final double[][] coef;
    private final double[][] linCoef;

    private double eta1;
    private double eta2;

    private boolean isFitted;

    public ANFIS(int numberOfRules, double eta1, double eta2) {
        this(numberOfRules, eta1, eta2, DEFAULT_LOSS_FUNCTION, DEFAULT_STOPPING_CONDITION);
    }

    public ANFIS(int numberOfRules, double eta1, double eta2, LossFunction lossFunction, StoppingCondition stoppingCondition) {
        this.numberOfRules = numberOfRules;
        this.lossFunction = lossFunction;
        this.stoppingCondition = stoppingCondition;

        this.eta1 = eta1;
        this.eta2 = eta2;

        coef = new double[numberOfRules][];
        linCoef = new double[numberOfRules][];
    }

    public double[][] getCoef() {
        if (!isFitted) throw new ModelNotFittedException(getClass());
        return coef;
    }

    public double[][] getLinCoef() {
        if (!isFitted) throw new ModelNotFittedException(getClass());
        return linCoef;
    }

    public double getEta1() {
        return eta1;
    }

    public void setEta1(double eta1) {
        this.eta1 = eta1;
    }

    public double getEta2() {
        return eta2;
    }

    public void setEta2(double eta2) {
        this.eta2 = eta2;
    }

    @Override
    public final IncrementalMachineLearningModel fit(List<Sample> samples) {
        checkEqualDimensions(samples);
        dimension = samples.get(0).x().length;
        initialize(coef, linCoef, dimension);
        isFitted = true;

        return train(samples);
    }

    private void checkEqualDimensions(List<Sample> samples) {
        int inputDimension = samples.get(0).x().length;
        if (!samples.stream().allMatch(sample -> sample.x().length == inputDimension && sample.y().length == 1))
            throw new InvalidDatasetException();
    }

    private ANFIS train(List<Sample> samples) {
        int iter = 0;
        while (true) {
            IterationStatistics statistics = IterationStatistics.of(() -> lossFunction.score(this, samples), iter);

            if (stoppingCondition.isMet(statistics)) break;

            notifyObservers(statistics);

            preprocess(samples);
            completeEpoch(samples);

            iter++;
        }

        return this;
    }

    protected void preprocess(List<Sample> samples) {
    }

    private void completeEpoch(List<Sample> samples) {
        for (List<Sample> batch : partition(samples)) {
            processBatch(batch);
        }
    }

    protected abstract List<List<Sample>> partition(List<Sample> samples);

    protected final void processBatch(List<Sample> batch) {
        double[][] dCoef = new double[numberOfRules][];
        double[][] dLinCoef = new double[numberOfRules][];
        construct(dCoef, dLinCoef, dimension);

        for (Sample sample : batch) {
            double[] input = sample.x();

            double target = sample.y()[0];
            double output = forwardPass(input)[0];
            double error = target - output;

            double weightSumSquared = cachedWeightSum * cachedWeightSum;
            double unscaledOutput = output * cachedWeightSum;

            for (int i = 0; i < numberOfRules; i++) {
                double weight = cachedWeights[i];
                double consequent = cachedConsequents[i];

                double fraction = ((cachedWeightSum - weight) * consequent - (unscaledOutput - weight * consequent)) / weightSumSquared;
                double dLinCoefShared = error * (weight / cachedWeightSum);
                for (int j = 0; j < dimension; j++) {
                    double membership = cachedMemberships[i][j];
                    double dCoefShared = error * fraction * (weight / membership) * membership * (1 - membership);
                    dCoef[i][2 * j] -= dCoefShared * (input[j] - coef[i][2 * j + 1]);
                    dCoef[i][2 * j + 1] += dCoefShared * coef[i][2 * j];
                    dLinCoef[i][j] += dLinCoefShared * input[j];
                }
                dLinCoef[i][dimension] += dLinCoefShared;
            }
        }

        for (int i = 0; i < numberOfRules; i++) {
            for (int j = 0; j < dimension; j++) {
                coef[i][2 * j] += eta1 * dCoef[i][2 * j];
                coef[i][2 * j + 1] += eta2 * dCoef[i][2 * j + 1];
                linCoef[i][j] += eta1 * dLinCoef[i][j];
            }
            linCoef[i][dimension] += eta1 * dLinCoef[i][dimension];
        }
    }

    @Override
    public final IncrementalMachineLearningModel partialFit(List<Sample> samples) {
        if (!isFitted) return fit(samples);
        checkCorrectDimension(samples);
        return train(samples);
    }

    private void checkCorrectDimension(List<Sample> samples) {
        if (dimension == 0) return;
        if (!samples.stream().allMatch(sample -> sample.x().length == dimension && sample.y().length == 1))
            throw new InvalidDatasetException();
    }

    @Override
    public final double[] predict(double[] input) {
        if (!isFitted) throw new ModelNotFittedException(getClass());

        if (input.length != dimension) throw new InputDimensionMismatch(dimension, input.length);

        return forwardPass(input);
    }

    private double[][] cachedMemberships;
    private double[] cachedWeights;
    private double cachedWeightSum = Double.NaN;
    private double[] cachedConsequents;

    private double[] forwardPass(double[] input) {
        double prediction = 0.;
        double weightsSum = 0.;
        double[][] memberships = new double[numberOfRules][];
        double[] weights = new double[numberOfRules];
        double[] consequents = new double[numberOfRules];
        for (int i = 0; i < numberOfRules; i++) {
            memberships[i] = new double[dimension];

            double weight = 1.;
            for (int j = 0; j < dimension; j++) {
                double membership = sigmoid(input[j], coef[i][2 * j], coef[i][2 * j + 1]);
                if (Double.isNaN(membership)) throw new CriticalDivergenceException(getClass());
                memberships[i][j] = membership;
                weight *= membership;
            }
            double consequent = linear(input, linCoef[i]);

            prediction += weight * consequent;
            weightsSum += weight;

            weights[i] = weight;
            consequents[i] = consequent;
        }

        cachedMemberships = memberships;
        cachedWeights = weights;
        cachedWeightSum = weightsSum;
        cachedConsequents = consequents;

        return new double[]{prediction / weightsSum};
    }

    private static double sigmoid(double input, double bi, double ai) {
        return 1. / (1. + Math.exp(bi * (input - ai)));
    }

    private static double linear(double[] input, double[] coefficients) {
        double value = 0;
        int dimension = coefficients.length - 1;
        for (int i = 0; i < dimension; i++) {
            value += input[i] * coefficients[i];
        }
        value += coefficients[dimension];
        return value;
    }

    private static final String COEF_DELIMITER = ";";

    public void save(String path) throws IOException {
        try (var writer = Files.newBufferedWriter(Path.of(path))) {
            for (int i = 0; i < numberOfRules; i++) {
                String coefString = Arrays.stream(coef[i]).mapToObj(String::valueOf).collect(Collectors.joining(COEF_DELIMITER));
                String linCoefString = Arrays.stream(linCoef[i]).mapToObj(String::valueOf).collect(Collectors.joining(COEF_DELIMITER));
                writer.write(coefString + COEF_DELIMITER + linCoefString);
                writer.newLine();
            }
        }
    }

    private static void construct(double[][] coef, double[][] linCoef, int dimension) {
        int coefCountPerDimension = 2 * dimension;
        int linCoefCountPerDimension = dimension + 1;
        for (int i = 0, n = coef.length; i < n; i++) {
            coef[i] = new double[coefCountPerDimension];
            linCoef[i] = new double[linCoefCountPerDimension];
        }
    }

    private static void initialize(double[][] coef, double[][] linCoef, int dimension) {
        construct(coef, linCoef, dimension);
        Random random = SourceOfRandomness.getSource();
        for (int i = 0, n = coef.length; i < n; i++) {
            for (int j = 0; j < dimension; j++) {
                coef[i][2 * j] = -0.5 + random.nextDouble();
                coef[i][2 * j + 1] = -0.5 + random.nextDouble();
                linCoef[i][j] = -0.5 + random.nextDouble();
            }
            linCoef[i][dimension] = 0.;
        }
    }
}
