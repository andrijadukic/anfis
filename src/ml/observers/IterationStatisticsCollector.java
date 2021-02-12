package ml.observers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class IterationStatisticsCollector implements MachineLearningModelObserver {

    private final List<IterationStatistics> collection;

    public IterationStatisticsCollector() {
        collection = new ArrayList<>();
    }

    @Override
    public void update(IterationStatistics statistics) {
        collection.add(statistics);
    }

    public void clear() {
        collection.clear();
    }

    public List<IterationStatistics> getCollection() {
        return Collections.unmodifiableList(collection);
    }
}
