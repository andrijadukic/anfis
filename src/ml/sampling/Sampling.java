package ml.sampling;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public final class Sampling {

    private Sampling() {
    }

    public static void write(List<Sample> samples, String path) throws IOException {
        write(samples, Sample::toString, path);
    }

    public static <T> void write(List<T> samples, Function<T, String> formatter, String path) throws IOException {
        try (var writer = Files.newBufferedWriter(Path.of(path))) {
            for (T sample : samples) {
                writer.write(formatter.apply(sample));
                writer.newLine();
            }
        }
    }

    public static List<Sample> load(String path) throws IOException {
        return load(path, Sample::parse);
    }

    public static <T> List<T> load(String path, Function<String, T> parser) throws IOException {
        return Files.lines(Path.of(path)).map(parser).collect(Collectors.toUnmodifiableList());
    }

    public static <T> List<List<T>> singletons(List<T> samples) {
        return new AbstractList<>() {
            @Override
            public List<T> get(int index) {
                return Collections.singletonList(samples.get(index));
            }

            @Override
            public int size() {
                return samples.size();
            }
        };
    }

    public static <T> List<List<T>> partition(List<T> samples, int partitionSize) {
        return new AbstractList<>() {

            private final int sampleSize = samples.size();
            private final int numberOfPartitions = Math.toIntExact(Math.round(Math.ceil((double) sampleSize / partitionSize)));

            @Override
            public List<T> get(int index) {
                final int start = index * partitionSize;
                final int end = Math.min(start + partitionSize, sampleSize);
                return samples.subList(start, end);
            }

            @Override
            public int size() {
                return numberOfPartitions;
            }
        };
    }
}
