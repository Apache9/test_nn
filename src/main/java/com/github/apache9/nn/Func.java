package com.github.apache9.nn;

public interface Func {

  double compute(double x);

  double differentiate(double x);
}
