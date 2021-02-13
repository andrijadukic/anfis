package anfis.ml.sampling;

import java.util.Random;

public class SourceOfRandomness {

    private static final Random random = new Random(System.nanoTime());

    public static Random getSource() {
        return random;
    }
}
