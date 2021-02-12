package ml.observers;

public final class NthIterationObserver implements MachineLearningModelObserver {

    private final MachineLearningModelObserver observer;
    private final int step;

    public NthIterationObserver(MachineLearningModelObserver observer, int step) {
        this.observer = observer;
        this.step = step;
    }

    @Override
    public void update(IterationStatistics statistics) {
        if (statistics.getIteration() % step == 0) {
            observer.update(statistics);
        }
    }
}
