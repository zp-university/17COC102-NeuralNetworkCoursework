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
    private final double learningRate;
    private final int maxEpoch;
    private final double maxError;


    /**
     * Creates a new NeuralNetwork object.
     *
     * @param learningRate  The learning rate value for this neural network
     * @param inputLayer    The amount of input neurons for the network
     * @param hiddenLayers  The amount of neurons in each hidden layer for the network
     * @param outputLayer   The amount of output neurons for the network
     */
    public NeuralNetwork(double learningRate, int maxEpoch, double maxError, int inputLayer, int[] hiddenLayers, int outputLayer) {
        this.learningRate = learningRate;
        this.inputLayer = inputLayer;
        this.hiddenLayers = hiddenLayers;
        this.outputLayer = outputLayer;
        this.maxEpoch = maxEpoch;
        this.maxError = maxError;

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
        biasNeuron.calculateProcessedOutput();
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

    public void runBackpropagation(double correctOutputs[]) {

        for(int i = neurons.size() - 1; i >= 1; i--) {
            LinkedList<Neuron> layer = neurons.get(i);
            for(int j = 0; j < layer.size(); ++j) {
                Neuron neuron = layer.get(j);
                double deltaValue = neuron.calculateDerivativeOutput(neuron.getProcessedOutput());
                //If this is the output layer, apply a slightly different function to get the delta value
                if(i == neurons.size() - 1) {
                    deltaValue = deltaValue * (correctOutputs[j] - neuron.getProcessedOutput());
                } else {
                    double weightErrorSum = 0;
                    for(Neuron rightLayerNeuron : neurons.get(i + 1)) {
                        Connection connection = rightLayerNeuron.getConnections().get(neuron);
                        weightErrorSum += connection.getWeight() * rightLayerNeuron.getDeltaValue();
                    }
                    deltaValue = deltaValue * weightErrorSum;
                }
                neuron.setDeltaValue(deltaValue);
                for(Connection connection : neuron.getConnections().values()) {
                    connection.setWeight(connection.getWeight() + (learningRate * neuron.getDeltaValue() * connection.getFromNeuron().getProcessedOutput()));
                }
            }
        }
    }

    public FinishReason runTraining() {
        //Run until maxEpoch, or forever if -1 is specified
        double totalError = 1;
        for(int i = 0; i != maxEpoch; ++i) {
            if(totalError <= maxError) {
                return FinishReason.BELOW_MAX_ERROR;
            }

            totalError = 0;

            for(int j = 0; j < inputs.length; ++j) {
                setInput(inputs[j]);
                runForwardOperation();

                finalOutput = getOutputs();

                for(int k = 0; k < correctOutputs[j].length; ++k) {
                    double inputError = Math.pow(finalOutput[k] - correctOutputs[j][k], 2);
                    totalError += inputError;
                }

                runBackpropagation(correctOutputs[0]);
            }
        }
        return FinishReason.REACHED_MAX_EPOCH;
    }

    public double getRandomWeight() {
        return ThreadLocalRandom.current().nextDouble(getRandomWeightMin(), getRandomWeightMax());
    }

    public enum FinishReason {
        REACHED_MAX_EPOCH,
        BELOW_MAX_ERROR
    }
}
