package pro.zackpollard.university.aicoursework;

import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import com.opencsv.CSVWriter;

import java.io.FileReader;
import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;

public class AICoursework {

    public static void main(String args[]) throws IOException {
        System.out.println("Hello world");

        CSVReader reader = new CSVReaderBuilder(new FileReader("../Data/data.csv"))
                .withSkipLines(1)
                .withCSVParser(new CSVParserBuilder()
                        .withSeparator(',')
                        .build()
                ).build();

        List<String[]> lines = reader.readAll();
        Collections.shuffle(lines);

        double[][] inputs = new double[lines.size()][];
        double[][] outputs = new double[lines.size()][];

        int count = 0;
        for(String[] record : lines) {
            double[] input = new double[record.length - 1];
            for(int i = 0; i < record.length - 1; ++i) {
                input[i] = Double.parseDouble(record[i]);
            }
            inputs[count] = input;
            outputs[count] = new double[]{Double.parseDouble(record[record.length - 1])};
            ++count;
        }

        NeuralNetwork neuralNetwork = new NeuralNetwork(0.2, 10000, 0.0001, 0.9, inputs, outputs,6, new int[]{6, 6}, 1);
        System.out.println(neuralNetwork.runTraining());
        System.out.println(neuralNetwork.getOutputs()[0]);
        double[][] comparison = new double[neuralNetwork.getTestInputs().length][2];
        double[][] output = neuralNetwork.runSeperateDataSet(neuralNetwork.getTestInputs());
        System.out.println("Test MSE = " + neuralNetwork.getMeanSquaredError(output, neuralNetwork.getCorrectTestOutputs()));
        neuralNetwork.denormalise(output);
        neuralNetwork.denormalise(neuralNetwork.getCorrectTestOutputs());

        for(int i = 0; i < output.length; ++i) {
            comparison[i][0] = output[i][0];
            comparison[i][1] = neuralNetwork.getCorrectTestOutputs()[i][0];
        }

        comparison = new double[neuralNetwork.getOriginalInputs().length][2];
        output = neuralNetwork.runSeperateDataSet(neuralNetwork.getOriginalInputs());
        System.out.println("Total MSE = " + neuralNetwork.getMeanSquaredError(output, neuralNetwork.getCorrectOriginalOutputs()));
        neuralNetwork.denormalise(output);
        neuralNetwork.denormalise(neuralNetwork.getCorrectOriginalOutputs());

        for(int i = 0; i < output.length; ++i) {
            comparison[i][0] = output[i][0];
            comparison[i][1] = neuralNetwork.getCorrectOriginalOutputs()[i][0];
        }

        Writer writer = Files.newBufferedWriter(Paths.get("../Data/output.csv"));
        CSVWriter csvWriter = new CSVWriter(writer, CSVWriter.DEFAULT_SEPARATOR, CSVWriter.NO_QUOTE_CHARACTER, CSVWriter.DEFAULT_ESCAPE_CHARACTER, CSVWriter.DEFAULT_LINE_END);
        csvWriter.writeNext(new String[]{"Actual Output", "Expected Output"});

        for(int i = 0; i < comparison.length; ++i) {
            csvWriter.writeNext(new String[]{String.valueOf(comparison[i][0]), String.valueOf(comparison[i][1])});
        }

        csvWriter.flush();
        writer.flush();

        System.out.println("Woop! :D");
    }
}
