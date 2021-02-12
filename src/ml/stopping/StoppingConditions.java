package ml.stopping;

public final class StoppingConditions {

    public static StoppingCondition precision(double epsilon) {
        return statistics -> statistics.getError() < epsilon;
    }

    public static StoppingCondition maxIter(int maxIter) {
        return statistics -> statistics.getIteration() > maxIter;
    }

    public static StoppingCondition infiniteLoop() {
        return statistics -> false;
    }
}
