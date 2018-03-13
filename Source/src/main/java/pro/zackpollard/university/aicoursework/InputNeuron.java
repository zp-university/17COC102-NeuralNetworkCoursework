package pro.zackpollard.university.aicoursework;

public class InputNeuron extends Neuron {

    @Override
    public double calculateProcessedOutput() {
        setProcessedOutput(getRawOutput());
        return getProcessedOutput();
    }
}
