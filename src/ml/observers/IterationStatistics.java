package ml.observers;

public interface IterationStatistics {

    double getError();

    int getIteration();

    static IterationStatistics of(LossFunctionScoreExtractor extractor, int iteration) {
        return new IterationStatistics() {

            private double error;
            private boolean isExtracted;

            @Override
            public double getError() {
                if (!isExtracted) {
                    error = extractor.extract();
                    isExtracted = true;
                }
                return error;
            }

            @Override
            public int getIteration() {
                return iteration;
            }
        };
    }

    static IterationStatistics of(double error, int iteration) {
        return new IterationStatistics() {
            @Override
            public double getError() {
                return error;
            }

            @Override
            public int getIteration() {
                return iteration;
            }
        };
    }

    interface LossFunctionScoreExtractor {

        double extract();
    }
}

