package pro.zackpollard.university.aicoursework;

public class AICoursework {

    public static void main(String args[]) {
        System.out.println("Hello world");
        NeuralNetwork neuralNetwork = new NeuralNetwork(0.1, 1000000, 0.000000001, 2, new int[]{2}, 1);
        System.out.println(neuralNetwork.runTraining());
        System.out.println(neuralNetwork.getOutputs()[0]);
        System.out.println(neuralNetwork.runSeperateDataSet(new double[][]{{1, 0}})[0][0]);
    }
}
