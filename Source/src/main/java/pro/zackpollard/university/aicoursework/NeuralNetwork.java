package pro.zackpollard.university.aicoursework;

import lombok.Getter;
import lombok.ToString;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

@ToString
public class NeuralNetwork {

    //Utilities
    private Random random = new Random();

    //Input Data
    private double originalInputs[][];
    private double correctOriginalOutputs[][];

    private double inputs[][];
    private double correctOutputs[][];

    private double validationInputs[][];
    private double correctValidationOutputs[][];

    @Getter
    private double testInputs[][];
    @Getter
    private double correctTestOutputs[][];

    //Normalisation Values
    private double[][] inputNormalisations;
    private double[][] outputNormalisations;

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
    private double learningRate;
    private final int maxEpoch;
    private final double minError;
    private final double momentumFactor;

    /**
     * Creates a new NeuralNetwork object.
     *
     * @param learningRate  The learning rate value for this neural network
     * @param inputLayer    The amount of input neurons for the network
     * @param hiddenLayers  The amount of neurons in each hidden layer for the network
     * @param outputLayer   The amount of output neurons for the network
     */
    public NeuralNetwork(double learningRate, int maxEpoch, double minError, double momentumFactor, double[][] inputs, double[][] outputs, int inputLayer, int[] hiddenLayers, int outputLayer) {
        this.learningRate = learningRate;
        this.inputLayer = inputLayer;
        this.hiddenLayers = hiddenLayers;
        this.outputLayer = outputLayer;
        this.maxEpoch = maxEpoch;
        this.minError = minError;
        this.momentumFactor = momentumFactor;
        this.originalInputs = inputs;
        this.correctOriginalOutputs = outputs;

        this.setup();
    }

    /**
     * Sets up the network based on the parameters specified by the user
     */
    private void setup() {

        System.out.println("Calculating min & max values...");
        //Normalise inputs and outputs
        inputNormalisations = new double[originalInputs[0].length][2];
        outputNormalisations = new double[correctOriginalOutputs[0].length][2];

        //Calculate min and max for inputs
        calculateMinMaxValues(inputLayer, originalInputs, inputNormalisations);
        //Calculate min and max for outputs
        calculateMinMaxValues(outputLayer, correctOriginalOutputs, outputNormalisations);

        System.out.println("Normalising...");

        //Normalise inputs
        normalise(originalInputs, inputNormalisations);
        //Normalise outputs
        normalise(correctOriginalOutputs, outputNormalisations);

        //End index of validation set of data
        int validationEnd = (originalInputs.length / 10) * 2;
        //End index of test set of data
        int testEnd = validationEnd * 2;
        //Split data into validation, test and training sets of inputs and outputs
        validationInputs = Arrays.copyOfRange(originalInputs, 0, validationEnd);
        correctValidationOutputs = Arrays.copyOfRange(correctOriginalOutputs, 0, validationEnd);
        testInputs = Arrays.copyOfRange(originalInputs, validationEnd, testEnd);
        correctTestOutputs = Arrays.copyOfRange(correctOriginalOutputs, validationEnd, testEnd);
        inputs = Arrays.copyOfRange(originalInputs, testEnd, originalInputs.length);
        correctOutputs = Arrays.copyOfRange(correctOriginalOutputs, testEnd, originalInputs.length);

        //Add the amount of InputNeuron's specified by the user to the input layer
        LinkedList<Neuron> inputNeurons = new LinkedList<>();
        neurons.add(0, inputNeurons);
        for(int i = 0; i < inputLayer; ++i) {
            inputNeurons.add(i, new InputNeuron());
        }

        //Add the amount of SigmoidNeuron's specified by the user to the hidden layer(s)
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

        //Add the amount of SigmoidNeuron's specified by the user to the output layer
        LinkedList<Neuron> outputNeurons = new LinkedList<>();
        neurons.add(neurons.size(), outputNeurons);
        for(int i = 0; i < outputLayer; ++i) {
            Neuron neuron = new SigmoidNeuron();
            outputNeurons.add(i, neuron);
            neuron.setupInputConnections(neurons.get(neurons.size() - 2), this);
            neuron.setupBiasConnection(biasNeuron, this);
        }
    }

    private void calculateMinMaxValues(int neurons, double[][] values, double[][] storageLocation) {
        for(int i = 0; i < neurons; ++i) {
            int maxValId = 0;
            int minValId = 0;
            for(int j = 0; j < values.length; ++j) {
                if(values[j][i] > values[maxValId][i]) {
                    maxValId = j;
                } else if(values[j][i] < values[minValId][i]) {
                    minValId = j;
                }
            }
            storageLocation[i][0] = values[maxValId][i];
            storageLocation[i][1] = values[minValId][i];
        }
    }

    private void normalise(double[][] values, double[][] minMaxStorageLocation) {
        for(int i = 0; i < values.length; ++i) {
            for(int j = 0; j < values[i].length; ++j) {
                double maxVal = minMaxStorageLocation[j][0];
                double minVal = minMaxStorageLocation[j][1];
                values[i][j] = ((values[i][j] - minVal) / (maxVal - minVal));
            }
        }
    }

    //This takes the noramlised set of outputs and edits them in-place to the normalised versions
    public void denormalise(double[][] output) {
        for(int i = 0; i < output.length; ++i) {
            for(int j = 0; j < output[i].length; ++j) {
                double maxVal = outputNormalisations[j][0];
                double minVal = outputNormalisations[j][1];
                output[i][j] = (maxVal - minVal) * ((output[i][j])) + minVal;
            }
        }
    }

    /**
     *
     * @param inputs
     */
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
                    double previousWeight = connection.getWeight();
                    double newWeight = connection.getWeight() + (learningRate * neuron.getDeltaValue() * connection.getFromNeuron().getProcessedOutput()) + (momentumFactor * connection.getPreviousWeightChange());
                    connection.setWeight(newWeight);
                    connection.setPreviousWeightChange(newWeight - previousWeight);
                }
            }
        }
    }

    public FinishReason runTraining() {
        //Run until maxEpoch, or forever if -1 is specified
        double previousValidationError = Double.MAX_VALUE;
        double totalError = 1;
        double lastError = 0;
        for(int epoch = 0; epoch != maxEpoch; ++epoch) {
            if(totalError <= minError) {
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

                    double exampleError = 0;
                    for(int j = 0; j < correctValidationOutputs[i].length; ++j) {
                        double outputError = Math.pow(output[j] - correctValidationOutputs[i][j], 2);
                        exampleError += outputError;
                    }
                    exampleError += exampleError / correctValidationOutputs[i].length;
                    validationError += exampleError;
                }

                validationError = validationError / validationInputs.length;
                System.out.println("Validation Error: " + validationError);
                System.out.println("Previous Validation Error: " + previousValidationError);

                if(validationError > previousValidationError) {
                    System.out.println(validationError);
                    System.out.println(previousValidationError);
                    restoreNetworkSnapshot(SnapshotName.LAST_VALIDATION_RUN);
                    return FinishReason.VALIDATION_ERROR_WORSE;
                }

                createNetworkSnapshot(SnapshotName.LAST_VALIDATION_RUN);
                previousValidationError = validationError;
            }

            //Create snapshot before processing is done for this epoch
            createNetworkSnapshot(SnapshotName.LAST_EPOCH);

            //Calculate error and do another round of backpropagation
            totalError = 0;
            for(int i = 0; i < inputs.length; ++i) {
                setInput(inputs[i]);
                runForwardOperation();

                double[] output = getOutputs();

                double exampleError = 0;
                for(int j = 0; j < correctOutputs[i].length; ++j) {
                    double inputError = Math.pow(correctOutputs[i][j] - output[j], 2);
                    exampleError += inputError;
                }
                exampleError = exampleError / correctOutputs[i].length;
                totalError += exampleError;

                runBackpropagation(correctOutputs[i]);
            }
            totalError = totalError / correctOutputs.length;
            if(epoch % 100 == 0) {
                System.out.println("Total Error:" + totalError);
            }

            //Bold Driver
            //Don't run on first pass
            if(lastError != 0) {
                if(totalError < lastError) {
                    learningRate *= 1.1;
                } else if(totalError> lastError + Math.pow(10, -10)) {
                    learningRate *= 0.5;
                    restoreNetworkSnapshot(SnapshotName.LAST_EPOCH);
                    epoch = epoch - 1;
                    continue;
                }
            }
            lastError = totalError;
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
