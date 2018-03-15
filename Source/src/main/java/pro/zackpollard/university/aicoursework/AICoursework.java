package pro.zackpollard.university.aicoursework;

import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;

import java.io.FileReader;
import java.io.IOException;
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

        NeuralNetwork neuralNetwork = new NeuralNetwork(0.2, 10000, 0.000001, 0.5, inputs, outputs,6, new int[]{6, 6}, 1);
        System.out.println(neuralNetwork.runTraining());
        System.out.println(neuralNetwork.getOutputs()[0]);
        double[][] comparison = new double[neuralNetwork.getTestInputs().length][2];
        double[][] output = neuralNetwork.runSeperateDataSet(neuralNetwork.getTestInputs());
        neuralNetwork.denormalise(output);
        neuralNetwork.denormalise(neuralNetwork.getCorrectTestOutputs());

        for(int i = 0; i < output.length; ++i) {
            comparison[i][0] = output[i][0];
            comparison[i][1] = neuralNetwork.getCorrectTestOutputs()[i][0];
        }

        System.out.println("Woop! :D");
    }
}
