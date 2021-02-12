package ml.sampling;

import java.util.Arrays;
import java.util.stream.Collectors;

public record Sample(double[] x, double[] y) {

    private static final String FEATURE_DELIMITER = ";";
    private static final String LABEL_DELIMITER = ";";
    private static final String FEATURE_LABEL_DELIMITER = ";";

    public Sample(double x, double y) {
        this(new double[]{x}, new double[]{y});
    }

    public Sample(double[] x, double y) {
        this(x, new double[]{y});
    }

    public Sample(double x, double[] y) {
        this(new double[]{x}, y);
    }

    @Override
    public String toString() {
        return Arrays.stream(x).mapToObj(String::valueOf).collect(Collectors.joining(FEATURE_DELIMITER)) +
                FEATURE_LABEL_DELIMITER +
                Arrays.stream(y).mapToObj(String::valueOf).collect(Collectors.joining(LABEL_DELIMITER));
    }

    public static Sample parse(String raw) {
        String[] sampleString = raw.split(FEATURE_LABEL_DELIMITER);
        double[] x = Arrays.stream(sampleString[0].split(FEATURE_DELIMITER)).mapToDouble(Double::parseDouble).toArray();
        double[] y = Arrays.stream(sampleString[1].split(LABEL_DELIMITER)).mapToDouble(Double::parseDouble).toArray();
        return new Sample(x, y);
    }
}
