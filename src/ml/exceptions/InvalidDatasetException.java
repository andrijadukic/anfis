package ml.exceptions;

public class InvalidDatasetException extends RuntimeException {

    public InvalidDatasetException() {
        super("Provided dataset is invalid");
    }

    public InvalidDatasetException(String message) {
        super("Provided dataset is invalid (" + message + ")");
    }
}
