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
    @Getter
    private double originalInputs[][];
    @Getter
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
     * @param learningRate      The learning rate value for this neural network
     * @param maxEpoch          The max amount of epochs that will run before the training terminates
     * @param minError          The minimum error at which the training will terminate
     * @param momentumFactor    The factor to apply to the momentum function
     * @param inputs            The inputs to run through the neural network
     * @param outputs           The correct outputs for the provided inputs
     * @param inputLayer        The amount of input neurons for the network
     * @param hiddenLayers      The amount of neurons in each hidden layer for the network
     * @param outputLayer       The amount of output neurons for the network
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

    //Calculates min and max values for a set of data and stores them in the storageLocation
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

    //Normalises all the values it is given based on the min and max values that are stored. This is all done in-place
    private void normalise(double[][] values, double[][] minMaxStorageLocation) {
        for(int i = 0; i < values.length; ++i) {
            for(int j = 0; j < values[i].length; ++j) {
                double maxVal = minMaxStorageLocation[j][0];
                double minVal = minMaxStorageLocation[j][1];
                values[i][j] = ((values[i][j] - minVal) / (maxVal - minVal));
            }
        }
    }

    //This takes the normalised set of outputs and edits them in-place to the normalised versions
    public void denormalise(double[][] output) {
        for(int i = 0; i < output.length; ++i) {
            for(int j = 0; j < output[i].length; ++j) {
                double maxVal = outputNormalisations[j][0];
                double minVal = outputNormalisations[j][1];
                output[i][j] = (maxVal - minVal) * ((output[i][j])) + minVal;
            }
        }
    }

    //Sets the inputs on the input neurons to the given values to prepare for a forward operation
    public void setInput(double inputs[]) {
        for(int i = 0; i < inputLayer; ++i) {
            neurons.getFirst().get(i).setRawOutput(inputs[i]);
        }
        biasNeuron.setRawOutput(1);
        biasNeuron.calculateProcessedOutput();
    }

    //Gets the current outputs on the output neurons from the last forward operation
    public double[] getOutputs() {
        double[] outputs = new double[outputLayer];
        for(int i = 0; i < outputLayer; ++i) {
            outputs[i] = neurons.getLast().get(i).getProcessedOutput();
        }
        return outputs;
    }

    //Runs through the network and computes the output values of each neuron
    public void runForwardOperation() {
        for(LinkedList<Neuron> layer : neurons) {
            for(Neuron neuron : layer) {
                neuron.calculateProcessedOutput();
            }
        }
    }

    //Runs a given data set on the trained network and returns all the outputs for the given inputs
    public double[][] runSeperateDataSet(double[][] inputs) {
        double[][] outputs = new double[inputs.length][outputLayer];
        for(int i = 0; i < inputs.length; ++i) {
            setInput(inputs[i]);
            runForwardOperation();
            outputs[i] = getOutputs();
        }
        return outputs;
    }

    //This runs the backpropagation function on the current network based on the given correct outputs
    public void runBackpropagation(double correctOutputs[]) {
        //Loops over all the layers in the network
        for(int i = neurons.size() - 1; i >= 1; i--) {
            //Gets the current layer
            LinkedList<Neuron> layer = neurons.get(i);
            //Loops over all the neurons in that layer
            for(int j = 0; j < layer.size(); ++j) {
                //Gets the current neuron
                Neuron neuron = layer.get(j);
                //Calculates the derivative output for that neuron
                double deltaValue = neuron.calculateDerivativeOutput(neuron.getProcessedOutput());
                //If this is the output layer, applies a slightly different function to get the delta value
                if(i == neurons.size() - 1) {
                    deltaValue = deltaValue * (correctOutputs[j] - neuron.getProcessedOutput());
                } else {
                    double weightErrorSum = 0;
                    //Loops over all connections to the right of the neuron and computes the sum of the weights * delta values
                    for(Neuron rightLayerNeuron : neurons.get(i + 1)) {
                        Connection connection = rightLayerNeuron.getConnections().get(neuron);
                        weightErrorSum += connection.getWeight() * rightLayerNeuron.getDeltaValue();
                    }
                    deltaValue = deltaValue * weightErrorSum;
                }
                //Sets the delta value on the neuron
                neuron.setDeltaValue(deltaValue);
                //Loops over all the connections to the left of that neuron and updates the weights
                for(Connection connection : neuron.getConnections().values()) {
                    double previousWeight = connection.getWeight();
                    //Calculates the new weight including the momentum from the last epoch
                    double newWeight = connection.getWeight() + (learningRate * neuron.getDeltaValue() * connection.getFromNeuron().getProcessedOutput()) + (momentumFactor * connection.getPreviousWeightChange());
                    connection.setWeight(newWeight);
                    connection.setPreviousWeightChange(newWeight - previousWeight);
                }
            }
        }
    }

    //Runs training and decides when to stop the network learning
    public FinishReason runTraining() {
        double previousValidationError = Double.MAX_VALUE;
        double totalError = 1;
        double lastError = 0;
        //Run until maxEpoch, or forever if -1 is specified
        for(int epoch = 0; epoch != maxEpoch; ++epoch) {
            //Checks if the error is less than the minimum error and stops learning if it is
            if(totalError <= minError) {
                restoreNetworkSnapshot(SnapshotName.LAST_EPOCH);
                return FinishReason.BELOW_MAX_ERROR;
            }

            //Validate neural network against validation set every 1000 epochs
            if(epoch % 1000 == 0) {
                System.out.println("Epoch: " + epoch);
                double validationError = 0;
                for(int i = 0; i < validationInputs.length; ++i) {
                    setInput(validationInputs[i]);
                    runForwardOperation();

                    double[] output = getOutputs();

                    double exampleError = 0;
                    //Loop over all correct outputs for this input set and calculate the sum of the errors
                    for(int j = 0; j < correctValidationOutputs[i].length; ++j) {
                        //Calculate the squared error for this output
                        double outputError = Math.pow(output[j] - correctValidationOutputs[i][j], 2);
                        exampleError += outputError;
                    }
                    //Divide the error by the amount of outputs to get the mean
                    exampleError += exampleError / correctValidationOutputs[i].length;
                    validationError += exampleError;
                }

                //Divide the overall error for the entire input set by the size of the input set to get the mean
                validationError = validationError / validationInputs.length;
                System.out.println("Validation Error: " + validationError);
                System.out.println("Previous Validation Error: " + previousValidationError);

                //Checks if the error is greater than the previous run of the validation set and stops learning
                //and resets to that previous state if it is
                if(validationError > previousValidationError) {
                    System.out.println(validationError);
                    System.out.println(previousValidationError);
                    restoreNetworkSnapshot(SnapshotName.LAST_VALIDATION_RUN);
                    return FinishReason.VALIDATION_ERROR_WORSE;
                }

                //Create a snapshot of this validation run for restoring on the next run if needed
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

            //Bold Driver calculation
            //Don't run on first pass
            /**if(lastError != 0) {
                //If error decreases, increase learningRate
                if(totalError < lastError) {
                    learningRate *= 1.1;
                //If error increases by anything over 10^-10 then halve learning rate and restore last epoch
                } else if(totalError> lastError + Math.pow(10, -10)) {
                    learningRate *= 0.5;
                    restoreNetworkSnapshot(SnapshotName.LAST_EPOCH);
                    epoch = epoch - 1;
                    continue;
                }
            }**/
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

    public double getMeanSquaredError(double[][] output, double[][] correctOutput) {
        double totalError = 0;
        for(int i = 0; i < output.length; ++i) {
            double exampleError = 0;
            for(int j = 0; j < correctOutput[i].length; ++j) {
                double inputError = Math.pow(correctOutput[i][j] - output[i][j], 2);
                exampleError += inputError;
            }
            exampleError = exampleError / correctOutput[i].length;
            totalError += exampleError;
        }
        totalError = totalError / correctOutputs.length;
        return totalError;
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
