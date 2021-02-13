package anfis.ml.exceptions;

public class InsufficientSampleSizeException extends RuntimeException {

    public InsufficientSampleSizeException(int expected, int actual) {
        super("Provided sample is not of sufficient size. Expected at least: " + expected + ", actual: " + actual);
    }
}
