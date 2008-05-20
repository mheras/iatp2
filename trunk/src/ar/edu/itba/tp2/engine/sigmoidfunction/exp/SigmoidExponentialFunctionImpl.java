package ar.edu.itba.tp2.engine.sigmoidfunction.exp;

import ar.edu.itba.tp2.engine.sigmoidfunction.SigmoidFunction;

public class SigmoidExponentialFunctionImpl implements SigmoidFunction {

	private double stepness;

	public SigmoidExponentialFunctionImpl(double stepness) {
		this.stepness = stepness;
	}

	public double derivativeOperate(double input) {
		return 2 * this.stepness * this.operate(input)
				* (1 - this.operate(input));
	}

	public double operate(double input) {
		return 1 / (1 + Math.exp((-2) * this.stepness * input));
	}

}
