package ar.edu.itba.tp2.engine.test;

import java.util.ArrayList;
import java.util.List;

import ar.edu.itba.tp2.engine.BackPropagation;
import ar.edu.itba.tp2.engine.configuration.BackPropagationConfiguration;
import ar.edu.itba.tp2.engine.function.Function;
import ar.edu.itba.tp2.engine.function.surfacefunction.SurfaceFunctionImpl;
import ar.edu.itba.tp2.engine.pattern.Pattern;
import ar.edu.itba.tp2.engine.pattern.PatternListFactory;
import ar.edu.itba.tp2.engine.pattern.PatternListFactoryConfiguration;
import ar.edu.itba.tp2.engine.sigmoidfunction.SigmoidFunction;
import ar.edu.itba.tp2.engine.sigmoidfunction.tanh.SigmoidTanHFunctionImpl;

public class FeedFowardPropagationTest {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		SigmoidFunction myFunction = new SigmoidTanHFunctionImpl(1);
		PatternListFactoryConfiguration myConfiguration = new PatternListFactoryConfiguration();
		myConfiguration.setQuantity(10);
		myConfiguration.setRealFunction(new SurfaceFunctionImpl());
		myConfiguration.setSigmoidFunction(myFunction);
		PatternListFactory patternFactory = new PatternListFactory();
		List<Pattern> myList = patternFactory.getRandomMeshPatternList(myConfiguration);
		
		
	
		
		/* Configuration of the Back Propagation Engine. */
		BackPropagationConfiguration currentConfiguration = new BackPropagationConfiguration();
		currentConfiguration.setActivationFunction(myFunction);
		currentConfiguration.setNInputs(2);
		currentConfiguration.setNOutputs(1);
		currentConfiguration.setNHiddenLayers(2);
		currentConfiguration.setNNeuronsInHiddenLayers(6);
		currentConfiguration.setLearningRate(0.05);
		currentConfiguration.setMaxEpochs(3000);
		currentConfiguration.setMomentum(0.000005);
		currentConfiguration.setAdaptableEtaAlpha(0.000005);
		currentConfiguration.setAdaptableEtaBeta(0.000005);
		/* This error its the BackPropagation Minimun Cuadratic Error for epoch. */
		currentConfiguration.setMinError(0.00000001);
		BackPropagation myBP = new BackPropagation(currentConfiguration);
		myBP.trainNeuralNetwork(myList);
		
		
		
		/* Ajustar el valor del error para asegurarse que porcentaje fue bien aprendido. */
		double averageLearned = myBP.checkLearnedPatterns(myList, 0.01);
		
		List test = patternFactory.getRandomMeshPatternList(myConfiguration);
		List result = myBP.testNeuralNetwork(test);
		List reMappedList = patternFactory.reMapList(myConfiguration, result);
		patternFactory.saveToFile(reMappedList, "prueba.out");
		
		/*Results Print.*/
		System.out.println();
		System.out.println("**************RESULTS***************************");
		System.out.println("Well Learned Patterns: "  + averageLearned*100 + "%");
		System.out.println("************************************************");
		return;
	}

}
