package anfis.ml.observers;

public final class NthIterationObserver implements ModelObserver {

    private final ModelObserver observer;
    private final int step;

    public NthIterationObserver(ModelObserver observer, int step) {
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
