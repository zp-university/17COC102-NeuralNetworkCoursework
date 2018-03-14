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

    private double validationInputs[][] = { {1, 0} };
    private double correctValidationOutputs[][] = { {1} };

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

    public double[][] runSeperateDataSet(double[][] inputs) {
        double[][] outputs = new double[inputs.length][outputLayer];
        for(int i = 0; i < inputs.length; ++i) {
            setInput(inputs[i]);
            runForwardOperation();
            outputs[i] = getOutputs();
        }
        return outputs;
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
        double previousValidationError = 1;
        double totalError = 1;
        for(int epoch = 0; epoch != maxEpoch; ++epoch) {
            if(totalError <= maxError) {
                restoreNetworkSnapshot(SnapshotName.LAST_EPOCH);
                return FinishReason.BELOW_MAX_ERROR;
            }

            //Run neural network against validation set every 1000 epochs
            if(epoch % 1000 == 0) {
                System.out.println("Epoch: " + epoch);
                double validationError = 0;
                for(int i = 0; i < validationInputs.length; ++i) {
                    setInput(validationInputs[i]);
                    runForwardOperation();

                    double[] output = getOutputs();

                    for(int j = 0; j < correctValidationOutputs[i].length; ++j) {
                        double inputError = Math.pow(output[j] - correctValidationOutputs[i][j], 2);
                        validationError += inputError;
                    }

                    if(validationError > previousValidationError) {
                        System.out.println(validationError);
                        System.out.println(previousValidationError);
                        restoreNetworkSnapshot(SnapshotName.LAST_VALIDATION_RUN);
                        return FinishReason.VALIDATION_ERROR_WORSE;
                    }

                    createNetworkSnapshot(SnapshotName.LAST_VALIDATION_RUN);
                    previousValidationError = validationError;
                }
            }

            //Calculate error and do another round of backpropagation
            totalError = 0;
            for(int i = 0; i < inputs.length; ++i) {
                createNetworkSnapshot(SnapshotName.LAST_EPOCH);
                setInput(inputs[i]);
                runForwardOperation();

                double[] output = getOutputs();

                for(int j = 0; j < correctOutputs[i].length; ++j) {
                    double inputError = Math.pow(output[j] - correctOutputs[i][j], 2);
                    totalError += inputError;
                }

                runBackpropagation(correctOutputs[0]);
            }
        }
        runForwardOperation();
        return FinishReason.REACHED_MAX_EPOCH;
    }

    private void restoreNetworkSnapshot(SnapshotName name) {
        for(LinkedList<Neuron> layer : neurons) {
            for(Neuron neuron : layer) {
                neuron.restoreSnapshot(name);
            }
        }
    }

    private void createNetworkSnapshot(SnapshotName name) {
        for(LinkedList<Neuron> layer : neurons) {
            for(Neuron neuron : layer) {
                neuron.createSnapshot(name);
            }
        }
    }

    public double getRandomWeight() {
        return ThreadLocalRandom.current().nextDouble(getRandomWeightMin(), getRandomWeightMax());
    }

    public enum FinishReason {
        REACHED_MAX_EPOCH,
        BELOW_MAX_ERROR,
        VALIDATION_ERROR_WORSE
    }

    public enum SnapshotName {
        LAST_EPOCH,
        LAST_VALIDATION_RUN
    }
}
