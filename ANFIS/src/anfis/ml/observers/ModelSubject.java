package anfis.ml.observers;

public interface ModelSubject {

    void addObserver(ModelObserver observer);

    void removeObserver(ModelObserver observer);

    void notifyObservers(IterationStatistics statistics);
}
