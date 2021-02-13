package anfis.ml.observers;

import java.util.ArrayList;
import java.util.List;

public class AbstractMachineLearningSubject implements MachineLearningModelSubject {

    private final List<MachineLearningModelObserver> observers;

    public AbstractMachineLearningSubject() {
        observers = new ArrayList<>();
    }

    @Override
    public void addObserver(MachineLearningModelObserver observer) {
        observers.add(observer);
    }

    @Override
    public void removeObserver(MachineLearningModelObserver observer) {
        observers.remove(observer);
    }

    @Override
    public void notifyObservers(IterationStatistics statistics) {
        observers.forEach(observer -> observer.update(statistics));
    }
}
