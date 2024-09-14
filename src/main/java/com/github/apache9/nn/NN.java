package com.github.apache9.nn;

import java.util.function.Function;

public class NN {

  private Layer head;

  private Layer tail;

  private final double eta;

  private final LossFunc lossFunc;

  public NN(double eta, LossFunc lossFunc) {
    this.eta = eta;
    this.lossFunc = lossFunc;
  }

  public void addHead(Layer layer) {
    this.head = layer;
    this.tail = layer;
  }

  public void addLayer(Function<Layer, Layer> creator) {
    this.tail = creator.apply(tail);
  }

  public void train(double[] input, double[] output) throws InterruptedException {
    assert input.length == head.getInputDim();
    assert output.length == tail.getOutputDim();
    for (int i = 0; ; i++) {
      forward(input);
      double loss = computeLoss(output);
      if (i % 100 == 0) {
        dump();
        System.out.printf("loss: %.04f%n", loss);
        System.out.println("=========================");
        Thread.sleep(1000);
      }

      if (loss < 1e-6) {
        break;
      }
      backpropagate(output);
    }
    forward(input);
    System.out.println("Final result:");
    dump();
  }

  private void forward(double[] input) {
    head.setInput(input);
    Layer l = head;
    while (l != null) {
      l.forward();
      l = l.getNext();
    }
  }

  private double computeLoss(double[] expected) {
    double[] output = tail.getActivation();
    double loss = 0;
    for (int i = 0; i < output.length; i++) {
      loss += lossFunc.compute(output[i], expected[i]);
    }
    return loss;
  }

  private void backpropagate(double[] expected) {
    double[] lossGradient = new double[expected.length];
    for (int i = 0; i < expected.length; i++) {
      lossGradient[i] = lossFunc.differentiate(tail.getActivation()[i], expected[i]);
    }
    Layer l = tail;
    while (l != null) {
      l.backpropagate(eta, lossGradient);
      l = l.getPrev();
    }
  }

  private void dump() {
    Layer l = head;
    while (l != null) {
      l.dump();
      l = l.getNext();
    }
  }

  public static void main(String[] args) throws Exception {
    NN nn = new NN(0.5, new MeanSquareLoss(2));
    nn.addHead(new Layer(3, 4, Sigmoid.INSTANCE));
    nn.addLayer(prev -> new Layer(prev, 2, Sigmoid.INSTANCE));
    nn.train(new double[] {0.3, 0.4, 0.5}, new double[] {0.8, 0.8});
  }
}
