package pro.zackpollard.university.aicoursework;

import lombok.Data;
import lombok.Getter;

import java.util.HashMap;

@Data
public class Connection {

    private double weight;
    private double previousWeightChange;

    private transient final Neuron fromNeuron;
    private transient final Neuron toNeuron;

    @Getter
    private HashMap<NeuralNetwork.SnapshotName, Double> snapshots = new HashMap<>();

    public void createSnapshot(NeuralNetwork.SnapshotName name) {
        this.snapshots.put(name, weight);
    }

    public double restoreSnapshot(NeuralNetwork.SnapshotName name) {
        this.weight = snapshots.get(name);
        return weight;
    }
}
