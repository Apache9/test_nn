package com.github.apache9.nn;

public interface LossFunc {

  double compute(double x, double expected);

  double differentiate(double x, double expected);
}
