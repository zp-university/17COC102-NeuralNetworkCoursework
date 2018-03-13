package pro.zackpollard.university.aicoursework;

public class AICoursework {

    public static void main(String args[]) {
        System.out.println("Hello world");
        NeuralNetwork neuralNetwork = new NeuralNetwork(0.01, 0.1, 2, new int[]{2}, 1);
        neuralNetwork.run();
    }
}
