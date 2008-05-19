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
		double[] inputs = { 0.0, 0.25};
		double[] outputs = { 0.5, 0.75};
		Pattern myPattern = new Pattern(inputs, outputs);

		Collection<Pattern> mySet = new HashSet<Pattern>();
		mySet.add(myPattern);

		SigmoidFunction myFunction = new SigmoidExponentialFunctionImpl(0.5);
		BackPropagation myBP = new BackPropagation(2, 2, 1, 2, 0.2, myFunction);
		myBP.trainNeuralNetwork(mySet);
	}

}
