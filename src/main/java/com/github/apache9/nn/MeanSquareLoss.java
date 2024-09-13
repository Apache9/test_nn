package com.github.apache9.nn;

public class MeanSquareLoss implements LossFunc {

  private final int dim;

  public MeanSquareLoss(int dim) {
    this.dim = dim;
  }

  @Override
  public double compute(double x, double expected) {
    return (x - expected) * (x - expected) / (2 * dim);
  }

  @Override
  public double differentiate(double x, double expected) {
    return (x - expected) / dim;
  }
}
