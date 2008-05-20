package ar.edu.itba.tp2.engine.test;

import java.util.Collection;
import java.util.HashSet;

import ar.edu.itba.tp2.engine.BackPropagation;
import ar.edu.itba.tp2.engine.pattern.Pattern;
import ar.edu.itba.tp2.engine.sigmoidfunction.SigmoidFunction;
import ar.edu.itba.tp2.engine.sigmoidfunction.exp.SigmoidExponentialFunctionImpl;

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
		inputs = new double[] { 0.0, 1.0 };
		outputs = new double[] { 1.0 };
		myPattern = new Pattern(inputs, outputs);
		mySet.add(myPattern);

		inputs = new double[] { 1.0, 0.0 };
		outputs = new double[] { 1.0 };
		myPattern = new Pattern(inputs, outputs);
		mySet.add(myPattern);

		inputs = new double[] { 1.0, 1.0 };
		outputs = new double[] { 0.0 };
		myPattern = new Pattern(inputs, outputs);
		mySet.add(myPattern);
		SigmoidFunction myFunction = new SigmoidExponentialFunctionImpl(0.5);
		BackPropagation myBP = new BackPropagation(2, 1, 0, 2, 0.1, 1000,
				myFunction);
		myBP.trainNeuralNetwork(mySet);
		inputs = new double[] { 1.0, 1.0 };
		System.out.println(myBP.testNeuralNetwork(inputs)[0]);
		return;
	}

}
