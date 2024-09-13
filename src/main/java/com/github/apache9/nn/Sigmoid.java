package com.github.apache9.nn;

public class Sigmoid implements Func {

  public static final Sigmoid INSTANCE = new Sigmoid();

  @Override
  public double compute(double x) {
    return 1.0 / (1.0 + Math.pow(Math.E, -x));
  }

  @Override
  public double differentiate(double x) {
    return compute(x) * (1 - compute(x));
  }
}
