package com.github.apache9.nn;

import java.util.Arrays;
import java.util.function.Function;

public class NN {

  private Layer head;

  private Layer tail;

  private final LossFunc lossFunc;

  public NN(LossFunc lossFunc) {
    this.lossFunc = lossFunc;
  }

  public void addHead(Layer layer) {
    this.head = layer;
    this.tail = layer;
  }

  public void addLayer(Function<Layer, Layer> creator) {
    this.tail = creator.apply(tail);
  }

  public void train(double[] input, double[] output) {
    assert input.length == head.getInputDim();
    assert output.length == tail.getOutputDim();
    for (; ; ) {
      forward(input);
      dump();
      double loss = computeLoss(output);
      System.out.printf("loss: %.04f%n", loss);
      System.out.println("=========================");
      if (loss < 0.01) {
        break;
      }
      backpropagate(output);
      try {
        Thread.sleep(5000);
      } catch (InterruptedException e) {
        throw new RuntimeException(e);
      }
    }
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
    double[] output = tail.getOutput();
    double loss = 0;
    for (int i = 0; i < output.length; i++) {
      loss  += lossFunc.compute(output[i], expected[i]);
    }
    return loss;
  }

  private void backpropagate(double[] expected) {

  }

  private void dump() {
    Layer l = head;
    while (l != null) {
      l.dump();
      l = l.getNext();
    }
  }

  public static void main(String[] args) {
    NN nn = new NN(new MeanSquareLoss(2));
    nn.addHead(new Layer(3, 4, Sigmoid.INSTANCE));
    nn.addLayer(prev -> new Layer(prev, 2, Sigmoid.INSTANCE));
    nn.train(new double[] {0.3, 0.4, 0.5}, new double[] {0.2, 0.2});
  }
}
