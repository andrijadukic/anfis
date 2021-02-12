package ml.exceptions;

public class InvalidSampleException extends RuntimeException {

    public InvalidSampleException() {
        super("Provided sample is invalid");
    }

    public InvalidSampleException(String message) {
        super("Provided sample is invalid (" + message + ")");
    }
}
