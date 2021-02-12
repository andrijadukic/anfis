package ml.exceptions;

public class InputDimensionMismatch extends RuntimeException {

    public InputDimensionMismatch(int expected, int actual) {
        super("Input is not of valid dimension. Expected: " + expected + ", actual: " + actual);
    }
}
