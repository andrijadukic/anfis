package ml.exceptions;

public class InputNotProcessedException extends RuntimeException {

    public InputNotProcessedException() {
        super("Input has not been processed yet");
    }
}
