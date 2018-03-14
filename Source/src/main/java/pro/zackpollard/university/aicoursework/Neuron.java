package pro.zackpollard.university.aicoursework;

import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

import java.util.LinkedHashMap;
import java.util.LinkedList;

@ToString(exclude={"connections"})
public abstract class Neuron {

    @Getter
    @Setter
    private double bias;
    @Getter
    @Setter
    private double rawOutput;
    @Getter
    @Setter
    private double processedOutput;
    @Getter
    @Setter
    private double deltaValue;

    @Getter
    private final LinkedHashMap<Neuron, Connection> connections = new LinkedHashMap<>();

    public void setupInputConnections(LinkedList<Neuron> neurons, NeuralNetwork instance) {
        for(Neuron neuron : neurons) {
            Connection connection = new Connection(neuron, this);
            connection.setWeight(instance.getRandomWeight());
            connections.put(neuron, connection);
        }
    }

    public void setupBiasConnection(Neuron biasNeuron, NeuralNetwork instance) {
        Connection connection = new Connection(biasNeuron, this);
        connection.setWeight(instance.getRandomWeight());
        connections.put(biasNeuron, connection);
    }

    public double calculateRawOutput() {
        double output = bias;
        for (Connection connection : connections.values()) {
            output += connection.getWeight() * connection.getFromNeuron().getProcessedOutput();
        }
        rawOutput = output;
        return output;
    }

    public abstract double calculateProcessedOutput();
    public abstract double calculateDerivativeOutput(double processedOutput);
}
