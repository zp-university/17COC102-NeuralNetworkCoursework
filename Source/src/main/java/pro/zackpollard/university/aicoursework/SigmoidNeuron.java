package pro.zackpollard.university.aicoursework;

public class SigmoidNeuron extends Neuron {

    @Override
    public double calculateProcessedOutput() {
        setProcessedOutput(sigmoid(calculateRawOutput()));
        return getProcessedOutput();
    }

    @Override
    public double calculateDerivativeOutput(double processedOutput) {
        return sigmoidDerivative(processedOutput);
    }

    private double sigmoid(double rawValue) {
        return 1 / (1 + Math.exp(-rawValue));
    }

    private double sigmoidDerivative(double rawValue) {
        return rawValue * (1 - rawValue);
    }
}
