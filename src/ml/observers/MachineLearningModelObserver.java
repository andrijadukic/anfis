package ml.observers;

public interface MachineLearningModelObserver {

    void update(IterationStatistics statistics);
}
