package pro.zackpollard.university.aicoursework;

public class SigmoidNeuron extends Neuron {

    @Override
    public double calculateProcessedOutput() {
        setProcessedOutput(sigmoid(calculateRawOutput()));
        return getProcessedOutput();
    }

    private double sigmoid(double rawValue) {
        return 1 / (1 + Math.exp(-rawValue));
    }
}
