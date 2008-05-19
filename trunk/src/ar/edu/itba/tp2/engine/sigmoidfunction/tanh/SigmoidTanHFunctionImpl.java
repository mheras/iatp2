package ar.edu.itba.tp2.engine.sigmoidfunction.tanh;

import ar.edu.itba.tp2.engine.sigmoidfunction.SigmoidFunction;

public class SigmoidTanHFunctionImpl implements SigmoidFunction{

	private double stepness;
	
	public SigmoidTanHFunctionImpl(double stepness){
		this.stepness = stepness;
	}
	
	public double derivativeOperate(double input) {
		return this.stepness * ( 1 - Math.pow(this.operate(input), 2));
	}

	public double operate(double input) {
		return Math.tanh(this.stepness * input);
	}

}
