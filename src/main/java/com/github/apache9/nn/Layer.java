package com.github.apache9.nn;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

public class Layer {

  private double[] input;

  private final Layer prev;

  private Layer next;

  private final int inputDim;

  private final int outputDim;

  private final Func activationFunc;

  private final double[][] weights;

  private final double[] output;

  private void fillWeights() {
    // random fill initial weights
    for (int i = 0; i < inputDim; i++) {
      for (int j = 0; j < outputDim; j++) {
        weights[i][j] = ThreadLocalRandom.current().nextDouble();
      }
    }
  }

  public Layer(int inputDim, int outputDim, Func activationFunc) {
    this.prev = null;
    this.inputDim = inputDim;
    this.outputDim = outputDim;
    this.activationFunc = activationFunc;
    this.weights = new double[inputDim][outputDim];
    this.output = new double[outputDim];
    fillWeights();
  }

  public Layer(Layer prev, int outputDim, Func activationFunc) {
    this.prev = prev;
    prev.next = this;
    this.inputDim = prev.outputDim;
    this.outputDim = outputDim;
    this.activationFunc = activationFunc;
    this.weights = new double[inputDim][outputDim];
    this.output = new double[outputDim];
    fillWeights();
  }

  private void forward(double[] input) {
    for (int i = 0; i < outputDim; i++) {
      double v = 0;
      for (int j = 0; j < inputDim; j++) {
        v += weights[j][i] * input[j];
      }
      output[i] = activationFunc.compute(v);
    }
  }

  public void forward() {
    forward(prev != null ? prev.output : input);
  }

  public void backpropagate(double eta, double[] lossGradient) {
    if (next != null) {

    } else {
      assert lossGradient.length == outputDim;
      for (int i = 0; i < lossGradient.length; i++) {
        lossGradient[i] *= activationFunc.differentiate(output[i]);
      }
      // compute gradient vector
      for (int j = 0; j < outputDim; j++) {

      }
      // update weight
      for (int i = 0; i < inputDim; i++) {
        for (int j = 0; j < outputDim; j++) {
          
        }
      }
    }
  }

  public void setInput(double[] input) {
    this.input = input;
  }

  public int getInputDim() {
    return inputDim;
  }

  public int getOutputDim() {
    return outputDim;
  }

  public double[] getOutput() {
    return output;
  }

  public Layer getNext() {
    return next;
  }

  public static void print(double[] v) {
    Arrays.stream(v).mapToObj(d -> String.format("%-8.04f", d)).forEach(System.out::print);
    System.out.println();
  }

  public void dump() {
    if (input != null) {
      print(input);
      System.out.println("=========================");
    }
    for (int i = 0; i < inputDim; i++) {
      print(weights[i]);
    }
    System.out.println("=========================");
    print(output);
    System.out.println("=========================");
  }
}
