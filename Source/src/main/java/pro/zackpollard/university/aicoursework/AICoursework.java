package pro.zackpollard.university.aicoursework;

public class AICoursework {

    public static void main(String args[]) {
        System.out.println("Hello world");
        NeuralNetwork neuralNetwork = new NeuralNetwork(0.1, 2, new int[]{10, 10, 10, 10, 10, 10, 10, 10}, 1);
        neuralNetwork.run();
    }
}
