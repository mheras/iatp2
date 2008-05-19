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
		double[] inputs = { 0.0, 0.1, 0.2, 0.3 };
		double[] outputs = { 1.0, 0.9, 0.8, 0.7 };
		Pattern myPattern = new Pattern(inputs, outputs);

		Collection<Pattern> mySet = new HashSet<Pattern>();
		mySet.add(myPattern);

		SigmoidFunction myFunction = new SigmoidExponentialFunctionImpl(0.5);
		BackPropagation myBP = new BackPropagation(4, 4, 1, 2, myFunction);
		myBP.trainNeuralNetwork(mySet);
	}

}
