package ar.edu.itba.tp2.engine.test;

import java.util.Collection;
import java.util.HashSet;

import ar.edu.itba.tp2.engine.BackPropagation;
import ar.edu.itba.tp2.engine.pattern.Pattern;
import ar.edu.itba.tp2.engine.sigmoidfunction.SigmoidFunction;
import ar.edu.itba.tp2.engine.sigmoidfunction.exp.SigmoidExponentialFunctionImpl;
import ar.edu.itba.tp2.engine.sigmoidfunction.tanh.SigmoidTanHFunctionImpl;

public class FeedFowardPropagationTest {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		double[] inputs = { 0.0, 0.0 };
		double[] outputs = { 0.0 };
		Pattern myPattern = new Pattern(inputs, outputs);

		Collection<Pattern> mySet = new HashSet<Pattern>();
		mySet.add(myPattern);
		double[] inputs2 = new double[] { 0.0, 1.0 };
		double[] outputs2 = new double[] { 1.0 };
		myPattern = new Pattern(inputs2, outputs2);
		mySet.add(myPattern);

		double[] inputs3 = new double[] { 1.0, 0.0 };
		double[] outputs3 = new double[] { 1.0 };
		myPattern = new Pattern(inputs3, outputs3);
		mySet.add(myPattern);

		double[] inputs4 = new double[] { 1.0, 1.0 };
		double[] outputs4 = new double[] { 0.0 };
		myPattern = new Pattern(inputs4, outputs4);
		mySet.add(myPattern);
		SigmoidFunction myFunction = new SigmoidTanHFunctionImpl(0.5);
		BackPropagation myBP = new BackPropagation(2, 1, 0, 2, 0.1, 1000,
				myFunction);
		myBP.trainNeuralNetwork(mySet);
		inputs = new double[] { 1.0, 1.0 };
		System.out.println(myBP.testNeuralNetwork(inputs)[0]);
		return;
	}

}
