package anfis.ml.observers;

import java.util.ArrayList;
import java.util.List;

public class AbstractModelSubject implements ModelSubject {

    private final List<ModelObserver> observers;

    public AbstractModelSubject() {
        observers = new ArrayList<>();
    }

    @Override
    public void addObserver(ModelObserver observer) {
        observers.add(observer);
    }

    @Override
    public void removeObserver(ModelObserver observer) {
        observers.remove(observer);
    }

    @Override
    public void notifyObservers(IterationStatistics statistics) {
        observers.forEach(observer -> observer.update(statistics));
    }
}
