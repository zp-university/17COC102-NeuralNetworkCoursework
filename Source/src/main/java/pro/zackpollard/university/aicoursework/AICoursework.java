package pro.zackpollard.university.aicoursework;

public class AICoursework {

    public static void main(String args[]) {
        System.out.println("Hello world");
        NeuralNetwork neuralNetwork = new NeuralNetwork(0.1, 1000000, 0.001, 2, new int[]{10, 10, 10, 10, 10, 10, 10, 10}, 1);
        System.out.println(neuralNetwork.runTraining());
        System.out.println(neuralNetwork.getOutputs()[0]);
    }
}
