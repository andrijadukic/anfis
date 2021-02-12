package ml.observers;

public interface MachineLearningModelSubject {

    void addObserver(MachineLearningModelObserver observer);

    void removeObserver(MachineLearningModelObserver observer);

    void notifyObservers(IterationStatistics statistics);
}
