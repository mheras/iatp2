package ar.edu.itba.tp2.engine.test;

import java.util.ArrayList;
import java.util.List;

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
		double[] inputs = { -1.0, -1.0 };
		double[] outputs = { -1.0 };
		Pattern myPattern = new Pattern(inputs, outputs);

		List<Pattern> myList = new ArrayList<Pattern>();
		myList.add(myPattern);
		double[] inputs2 = new double[] { -1.0, 1.0 };
		double[] outputs2 = new double[] { -1.0 };
		myPattern = new Pattern(inputs2, outputs2);
		myList.add(myPattern);

		double[] inputs3 = new double[] { 1.0, -1.0 };
		double[] outputs3 = new double[] { -1.0 };
		myPattern = new Pattern(inputs3, outputs3);
		myList.add(myPattern);

		double[] inputs4 = new double[] { 1.0, 1.0 };
		double[] outputs4 = new double[] { 1.0 };
		myPattern = new Pattern(inputs4, outputs4);
		myList.add(myPattern);
		SigmoidFunction myFunction = new SigmoidTanHFunctionImpl(2);
		BackPropagation myBP = new BackPropagation(2, 1, 1, 2, 0.1, 1200,
				myFunction);
		myBP.trainNeuralNetwork(myList);
		inputs = new double[] { 1.0, -1.0 };
		System.out.println("RESULT:");
		System.out.println(myBP.testNeuralNetwork(inputs)[0]);
		return;
	}

}
