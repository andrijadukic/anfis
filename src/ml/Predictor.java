package ml;

@FunctionalInterface
public interface Predictor {

    double[] predict(double[] x);

    default double[][] predict(double[]... data) {
        int n = data.length;
        double[][] predictions = new double[n][];
        for (int i = 0; i < n; i++) {
            predictions[i] = predict(data[i]);
        }
        return predictions;
    }
}
