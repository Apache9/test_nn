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

  private final double[] activation;

  private final double[] gradient;

  private void fillWeights() {
    // random fill initial weights
    for (int i = 0; i < outputDim; i++) {
      for (int j = 0; j < inputDim; j++) {
        weights[i][j] = ThreadLocalRandom.current().nextDouble();
      }
    }
  }

  public Layer(int inputDim, int outputDim, Func activationFunc) {
    this.prev = null;
    this.inputDim = inputDim;
    this.outputDim = outputDim;
    this.activationFunc = activationFunc;
    this.weights = new double[outputDim][inputDim];
    this.output = new double[outputDim];
    this.activation = new double[outputDim];
    this.gradient = new double[outputDim];
    fillWeights();
  }

  public Layer(Layer prev, int outputDim, Func activationFunc) {
    this.prev = prev;
    prev.next = this;
    this.inputDim = prev.outputDim;
    this.outputDim = outputDim;
    this.activationFunc = activationFunc;
    this.weights = new double[outputDim][inputDim];
    this.output = new double[outputDim];
    this.activation = new double[outputDim];
    this.gradient = new double[outputDim];
    fillWeights();
  }

  private void forward(double[] input) {
    for (int i = 0; i < outputDim; i++) {
      output[i] = 0;
      for (int j = 0; j < inputDim; j++) {
        output[i] += weights[i][j] * input[j];
      }
      activation[i] = activationFunc.compute(output[i]);
    }
  }

  public void forward() {
    forward(prev != null ? prev.output : input);
  }

  private void computeGradient(double[] lossGradient) {
    if (next != null) {
      for (int i = 0; i < outputDim; i++) {
        gradient[i] = 0;
        for (int j = 0; j < next.outputDim; j++) {
          gradient[i] += next.gradient[j] * next.weights[j][i] * activationFunc.compute(output[i]);
        }
      }
    } else {
      assert lossGradient.length == outputDim;
      for (int i = 0; i < lossGradient.length; i++) {
        gradient[i] = lossGradient[i] * activationFunc.differentiate(output[i]);
      }
    }
  }

  private void updateWeights(double eta, double[] input) {
    for (int i = 0; i < outputDim; i++) {
      for (int j = 0; j < inputDim; j++) {
        weights[i][j] -= eta * gradient[i] * input[j];
      }
    }
  }

  public void backpropagate(double eta, double[] lossGradient) {
    computeGradient(lossGradient);
    updateWeights(eta, prev != null ? prev.output : input);
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

  public double[] getActivation() {
    return activation;
  }

  public Layer getPrev() {
    return prev;
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
    for (int i = 0; i < outputDim; i++) {
      print(weights[i]);
    }
    System.out.println("=========================");
    print(output);
    print(activation);
    System.out.println("=========================");
  }
}
