package ar.edu.itba.tp2.engine.function.surfacefunction;

import ar.edu.itba.tp2.engine.function.Function;

public class SurfaceFunctionImpl implements Function {

	/* args[0] = x and args[1] = y. */
	public double[] operate(double[] args) {

		double[] result = new double[1];

		result[0] = args[0]*Math.exp(Math.pow(((-1) * args[0]), 2)
								- Math.pow(args[1], 2));
		return result;
	}

}
