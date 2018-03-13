package pro.zackpollard.university.aicoursework;

import lombok.Getter;
import lombok.ToString;

import java.util.LinkedList;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

@ToString
public class NeuralNetwork {

    //Utilities
    private Random random = new Random();

    //Test Input Data
    private double inputs[][] = { {1, 0} };
    private double correctOutputs[][] = { {1} };

    //Calculated outputs
    private double currentOutputs[][] = {};
    private double finalOutput[] = {};

    //Neurons
    private final LinkedList<LinkedList<Neuron>>  neurons = new LinkedList<>();
    private final int inputLayer;
    private final int[] hiddenLayers;
    private final int outputLayer;
    private final Neuron biasNeuron = new InputNeuron();

    //Values
    @Getter(lazy = true)
    private final double randomWeightMax = 2D / (double) inputLayer;
    @Getter(lazy = true)
    private final double randomWeightMin = -getRandomWeightMax();


    //Hyperparameters
    private final double epsilon;
    private final double learningRate;


    /**
     * Creates a new NeuralNetwork object.
     *
     * @param epsilon       The epsilon value for this neural network
     * @param learningRate  The learning rate value for this neural network
     * @param inputLayer    The amount of input neurons for the network
     * @param hiddenLayers  The amount of neurons in each hidden layer for the network
     * @param outputLayer   The amount of output neurons for the network
     */
    public NeuralNetwork(double epsilon, double learningRate, int inputLayer, int[] hiddenLayers, int outputLayer) {
        this.epsilon = epsilon;
        this.learningRate = learningRate;
        this.inputLayer = inputLayer;
        this.hiddenLayers = hiddenLayers;
        this.outputLayer = outputLayer;

        this.setup();
    }

    private void setup() {

        LinkedList<Neuron> inputNeurons = new LinkedList<>();
        neurons.add(0, inputNeurons);
        for(int i = 0; i < inputLayer; ++i) {
            inputNeurons.add(i, new InputNeuron());
        }

        for(int i = 0; i < hiddenLayers.length; ++i) {
            LinkedList<Neuron> hiddenNeurons = new LinkedList<>();
            neurons.add(i + 1, hiddenNeurons);
            for(int j = 0; j < hiddenLayers[i]; ++j) {
                Neuron neuron = new SigmoidNeuron();
                hiddenNeurons.add(j, neuron);
                neuron.setupInputConnections(neurons.get(i), this);
                neuron.setupBiasConnection(biasNeuron, this);
            }
        }

        LinkedList<Neuron> outputNeurons = new LinkedList<>();
        neurons.add(neurons.size(), outputNeurons);
        for(int i = 0; i < outputLayer; ++i) {
            Neuron neuron = new SigmoidNeuron();
            outputNeurons.add(i, neuron);
            neuron.setupInputConnections(neurons.get(neurons.size() - 2), this);
            neuron.setupBiasConnection(biasNeuron, this);
        }
    }

    public void setInput(double inputs[]) {
        for(int i = 0; i < inputLayer; ++i) {
            neurons.getFirst().get(i).setRawOutput(inputs[i]);
        }
        biasNeuron.setRawOutput(1);
        biasNeuron.setProcessedOutput(1);
    }

    public double[] getOutputs() {
        double[] outputs = new double[outputLayer];
        for(int i = 0; i < outputLayer; ++i) {
            outputs[i] = neurons.getLast().get(i).getProcessedOutput();
        }
        return outputs;
    }

    public void runForwardOperation() {
        for(LinkedList<Neuron> layer : neurons) {
            for(Neuron neuron : layer) {
                neuron.calculateProcessedOutput();
            }
        }
    }

    public void run() {
        setInput(inputs[0]);
        runForwardOperation();
        System.out.println(getOutputs()[0]);
    }

    public double getRandomWeight() {
        return ThreadLocalRandom.current().nextDouble(getRandomWeightMin(), getRandomWeightMax());
    }
}
