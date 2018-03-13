package pro.zackpollard.university.aicoursework;

import lombok.Data;

@Data
public class Connection {

    private double weight;
    private double deltaWeight;

    private transient final Neuron fromNeuron;
    private transient final Neuron toNeuron;
}
