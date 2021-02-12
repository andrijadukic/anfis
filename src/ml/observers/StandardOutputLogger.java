package ml.observers;

public final class StandardOutputLogger implements MachineLearningModelObserver {

    @Override
    public void update(IterationStatistics statistics) {
        System.out.println("Iteration: " + statistics.getIteration() + " | Error: " + statistics.getError());
    }
}
