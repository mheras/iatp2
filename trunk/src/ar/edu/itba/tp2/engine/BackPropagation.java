package ar.edu.itba.tp2.engine;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import ar.edu.itba.tp2.engine.configuration.BackPropagationConfiguration;
import ar.edu.itba.tp2.engine.exception.InvalidInputException;
import ar.edu.itba.tp2.engine.exception.InvalidTrainPatternException;
import ar.edu.itba.tp2.engine.function.Function;
import ar.edu.itba.tp2.engine.pattern.Pattern;
import ar.edu.itba.tp2.engine.sigmoidfunction.SigmoidFunction;

/**
 * @author Jorge Goldman
 * 
 * This class is a BackPropagtion Learning Neural Network Engine.
 * 
 */
public class BackPropagation {

	private SigmoidFunction activationFunction = null;

	private Integer nInputs;

	private Integer nOutputs;

	private Integer nHiddenLayers;

	private Integer nNeuronsInHiddenLayers;

	private double learningRate;

	private double[][][] weights;

	private double[][][] previousWeights = null;

	private long maxEpochs;

	private double momentum;

	private Perceptron[][] perceptronMatrix;

	private double currentError;

	private double previousError = Double.MAX_VALUE;

	private double adaptableEtaAlpha;

	private double adaptableEtaBeta;

	private double minError;

	public BackPropagation(BackPropagationConfiguration configuration) {
		this.nInputs = configuration.getNInputs();
		this.nOutputs = configuration.getNOutputs();
		this.nHiddenLayers = configuration.getNHiddenLayers();
		this.nNeuronsInHiddenLayers = configuration.getNNeuronsInHiddenLayers();
		this.activationFunction = configuration.getActivationFunction();
		this.learningRate = configuration.getLearningRate();
		this.maxEpochs = configuration.getMaxEpochs();
		/* If this parameter is 0, there is no momentum. */
		this.momentum = configuration.getMomentum();
		/* If this two parameters are 0, there is no adaptableEta improvement. */
		this.adaptableEtaAlpha = configuration.getAdaptableEtaAlpha();
		this.adaptableEtaBeta = configuration.getAdaptableEtaBeta();
		/* BackPropagation Minimum Cuadratic Error for Epoch Limit. */
		this.minError = configuration.getMinError();

		weights = new double[nHiddenLayers + 1][][];
		perceptronMatrix = new Perceptron[nHiddenLayers + 2][];
		currentError = 0;

		/* Stage 1: Initialize the weights. */
		this.initializePerceptrons();
		this.initializeWeightMatrix();
	}

	private void initializePerceptrons() {

		/*
		 * Iterates over the layers of the neural network, the layer 0 is the
		 * input layer. The N - 1 layer is the output layer.
		 */
		for (int m = 0; m < this.nHiddenLayers + 2; m++) {

			/* The input layer. */
			if (m == 0) {
				this.perceptronMatrix[m] = new Perceptron[this.nInputs + 1];
				for (int k = 0; k < this.nInputs + 1; k++) {
					this.perceptronMatrix[m][k] = new Perceptron();
				}
				/* The BIAS Input. */
				this.perceptronMatrix[m][this.nInputs].setOutput(-1);
			}
			/* The output layer. */
			else if (m == (this.nHiddenLayers + 1)) {
				this.perceptronMatrix[m] = new Perceptron[this.nOutputs];

				for (int i = 0; i < this.nOutputs; i++) {
					this.perceptronMatrix[m][i] = new Perceptron();
				}
			}
			/* The Intermediate Hidden Layers. */
			else {
				this.perceptronMatrix[m] = new Perceptron[this.nNeuronsInHiddenLayers + 1];
				for (int j = 0; j < this.nNeuronsInHiddenLayers + 1; j++) {
					this.perceptronMatrix[m][j] = new Perceptron();
				}
				/* The BIAS Input in the Hidden Layer i. */
				this.perceptronMatrix[m][this.nNeuronsInHiddenLayers]
						.setOutput(-1);
			}
		}
	}

	/**
	 * This method initialize the array of weight matrices.
	 */
	private void initializeWeightMatrix() {

		/* If there is no hidden layers... */
		if (this.nHiddenLayers == 0) {
			this.weights[0] = new double[this.nOutputs][this.nInputs + 1];
			for (int i = 0; i < this.nOutputs; i++) {
				for (int k = 0; k < this.nInputs + 1; k++) {
					this.weights[0][i][k] = randomWeight();
				}
			}
			return;
		}

		/* If there are hidden layers... */
		for (int m = 0; m < this.nHiddenLayers + 1; m++) {
			/* Weights between the input layer and the first hidden layer. */
			if (m == 0) {
				this.weights[m] = new double[this.nNeuronsInHiddenLayers][this.nInputs + 1];
				for (int j = 0; j < this.nNeuronsInHiddenLayers; j++) {
					for (int k = 0; k < this.nInputs + 1; k++) {
						this.weights[m][j][k] = randomWeight();
					}
				}
			}
			/* Weights between the last hidden layer and the output. */
			else if (m == this.nHiddenLayers) {
				this.weights[m] = new double[this.nOutputs][this.nNeuronsInHiddenLayers + 1];
				for (int i = 0; i < this.nOutputs; i++) {
					for (int j = 0; j < this.nNeuronsInHiddenLayers + 1; j++) {
						this.weights[m][i][j] = randomWeight();
					}
				}
			}
			/* Weights between hidden layers. */
			else {
				this.weights[m] = new double[this.nNeuronsInHiddenLayers][this.nNeuronsInHiddenLayers + 1];
				for (int j1 = 0; j1 < this.nNeuronsInHiddenLayers; j1++) {
					for (int j2 = 0; j2 < this.nNeuronsInHiddenLayers + 1; j2++) {
						this.weights[m][j1][j2] = randomWeight();
					}
				}
			}
		}
	}

	/**
	 * @see http://richardbowles.tripod.com/neural/backprop/backprop.htm
	 */
	public void trainNeuralNetwork(List<Pattern> trainPatternSet) {

		boolean giveMomentum = false;
		double newEta = this.learningRate;
		// this.printWeights();
		/* For every epoch... */
		double sumOfAllErrors = Double.MAX_VALUE;
		for (long epoch = 0L; (epoch < this.maxEpochs)
				&& (sumOfAllErrors > this.minError); epoch++) {
			/* We shuffle the patterns, in order to get best results. */
			Collections.shuffle(trainPatternSet);
			/* For every pattern... */
			sumOfAllErrors = 0;
			for (Pattern currentPattern : trainPatternSet) {

				double[] inputs = currentPattern.getInput();
				double[][] deltas = new double[this.nHiddenLayers + 2][];

				/*
				 * Checks the length of the inputs corresponds to the length of
				 * the neural network.
				 */
				if (inputs.length != this.nInputs) {
					throw new InvalidTrainPatternException();
				}

				/*
				 * Stage 2: Puts the input in the neural network. The input
				 * layer is the 0 layer.
				 */
				for (int k = 0; k < this.nInputs; k++) {
					this.perceptronMatrix[0][k].setOutput(inputs[k]);
				}

				/* Stage 3: Propagates the signal forwards throught the network. */
				this.propagateInputSignal();

				/* Stage 4: Compute deltas for the output layer. */
				deltas[this.nHiddenLayers + 1] = this
						.computeDeltasOutputLayer(currentPattern.getOutput());
				sumOfAllErrors += this.currentError;
				if (epoch != 0 && this.currentError < this.previousError) {
					giveMomentum = true;
					newEta += this.adaptableEtaAlpha;
				} else {
					giveMomentum = false;
					newEta += (-1) * this.adaptableEtaBeta * newEta;
				}
				this.previousError = this.currentError;
				/*
				 * Stage 5: Compute deltas for the preceding layers by
				 * propagating the errors backwards (not including the input
				 * layer).
				 */
				for (int m = this.nHiddenLayers; m > 0; m--) {
					deltas[m] = new double[this.nNeuronsInHiddenLayers];
					for (int i = 0; i < this.nNeuronsInHiddenLayers; i++) {
						double delta = 0;
						int upperLayerCount;

						/*
						 * If upper layer is a hidden layer, we need to ignore
						 * the BIAS perceptron.
						 */
						if (m == this.nHiddenLayers) {
							upperLayerCount = this.nOutputs;
						} else {
							upperLayerCount = this.nNeuronsInHiddenLayers;
						}

						for (int j = 0; j < upperLayerCount; j++) {
							delta += this.weights[m][j][i] * deltas[m + 1][j];
						}
						delta *= this.activationFunction
								.derivativeOperate(this.perceptronMatrix[m][i]
										.getH());
						deltas[m][i] = delta;
					}
				}

				/*
				 * Stage 6: Update all weights.
				 */

				/* There are hidden layers. */
				if (this.nHiddenLayers != 0) {
					for (int m = 0; m < this.nHiddenLayers + 1; m++) {
						/* Weigths between the input and first hidden layer. */
						if (m == 0) {
							for (int j = 0; j < this.nInputs + 1; j++) {
								for (int i = 0; i < this.nNeuronsInHiddenLayers; i++) {
									double addMomentum = 0;
									if (giveMomentum) {
										addMomentum = this.momentum
												* this.previousWeights[m][i][j];
									}
									/* adaptable Eta */
									this.weights[m][i][j] += newEta
											* deltas[m + 1][i]
											* this.perceptronMatrix[m][j]
													.getOutput() + addMomentum;
								}
							}
						}
						/* Weights between the output and hidden layers. */
						else if (m == this.nHiddenLayers) {
							for (int j = 0; j < this.nNeuronsInHiddenLayers + 1; j++) {
								for (int i = 0; i < this.nOutputs; i++) {
									double addMomentum = 0;
									if (giveMomentum) {
										addMomentum = this.momentum
												* this.previousWeights[m][i][j];
									}
									this.weights[m][i][j] += newEta
											* deltas[m + 1][i]
											* this.perceptronMatrix[m][j]
													.getOutput() + addMomentum;
								}
							}
						}
						/* Weights between hidden layers. */
						else {
							for (int j = 0; j < this.nNeuronsInHiddenLayers + 1; j++) {
								for (int i = 0; i < this.nNeuronsInHiddenLayers; i++) {
									double addMomentum = 0;
									if (giveMomentum) {
										addMomentum = this.momentum
												* this.previousWeights[m][i][j];
									}
									this.weights[m][i][j] += newEta
											* deltas[m + 1][i]
											* this.perceptronMatrix[m][j]
													.getOutput() + addMomentum;
								}
							}
						}
					}
				}
				/* There is no hidden layers. */
				else {
					for (int i = 0; i < this.nOutputs; i++) {
						for (int j = 0; j < this.nInputs + 1; j++) {
							double addMomentum = 0;
							if (giveMomentum) {
								addMomentum = this.momentum
										* this.previousWeights[0][i][j];
							}
							this.weights[0][i][j] += newEta * deltas[1][i]
									* this.perceptronMatrix[0][j].getOutput()
									+ addMomentum;
						}
					}
				}
				this.previousWeights = this.cloneWeights(weights);
			}

			// System.out.println("**********************");
			// System.out.println(String.format("Epoch %d", epoch));
			// this.printWeights();
			System.out.println(sumOfAllErrors);
		}
		// System.out.println();
		// this.printWeights();
	}

	public double[] testNeuralNetwork(double[] input) {
		if (input.length != this.nInputs) {
			throw new InvalidInputException();
		}
		for (int k = 0; k < this.nInputs; k++) {
			this.perceptronMatrix[0][k].setOutput(input[k]);
		}

		this.propagateInputSignal();

		double[] result = new double[this.nOutputs];
		for (int j = 0; j < this.nOutputs; j++) {
			result[j] = this.perceptronMatrix[this.nHiddenLayers + 1][j]
					.getOutput();
		}
		return result;
	}

	public List<Pattern> testNeuralNetwork(List<Pattern> inputList){
		List<Pattern> resultList = new ArrayList<Pattern>();
		
		for(Pattern currentPattern: inputList){
			double [] inputs = currentPattern.getInput();
			double [] outputs = this.testNeuralNetwork(inputs);
			Pattern newPattern = new Pattern(inputs, outputs);
			resultList.add(newPattern);
		}
		
		return resultList;
	}
	private double[] computeDeltasOutputLayer(double[] correctOutputs) {
		double[] result = new double[this.nOutputs];
		Perceptron[] NNoutputs = this.perceptronMatrix[this.nHiddenLayers + 1];
		double sum = 0;
		for (int i = 0; i < this.nOutputs; i++) {
			result[i] = this.activationFunction.derivativeOperate(NNoutputs[i]
					.getH())
					* (correctOutputs[i] - NNoutputs[i].getOutput());
			sum += (0.5) * Math.pow((correctOutputs[i] - NNoutputs[i]
					.getOutput()), 2);
		}
		this.currentError = sum;
		// System.out.println(sum);
		return result;
	}

	private void propagateInputSignal() {

		/* There are hidden layers. */
		if (this.nHiddenLayers != 0) {
			for (int m = 0; m < this.nHiddenLayers + 1; m++) {
				double[][] weightMatrix = this.weights[m];
				/* Between Inputs and Hidden Layers (BIAS included). */
				if (m == 0) {

					for (int j = 0; j < this.nNeuronsInHiddenLayers; j++) {
						double h = 0;
						for (int k = 0; k < this.nInputs + 1; k++) {
							h += weightMatrix[j][k]
									* this.perceptronMatrix[m][k].getOutput();
						}
						this.perceptronMatrix[m + 1][j].setH(h);
						this.perceptronMatrix[m + 1][j]
								.setOutput(this.activationFunction.operate(h));
					}
				}
				/* Between Hidden Layer and Outputs. Plus the BIAS. */
				else if (m == this.nHiddenLayers) {

					for (int i = 0; i < this.nOutputs; i++) {
						double h = 0;
						for (int j = 0; j < this.nNeuronsInHiddenLayers + 1; j++) {
							h += weightMatrix[i][j]
									* this.perceptronMatrix[m][j].getOutput();
						}
						this.perceptronMatrix[m + 1][i].setH(h);
						this.perceptronMatrix[m + 1][i]
								.setOutput(this.activationFunction.operate(h));
					}

				}
				/* Between hidden layers. */
				else {

					for (int j1 = 0; j1 < this.nNeuronsInHiddenLayers; j1++) {
						double h = 0;
						for (int j2 = 0; j2 < this.nNeuronsInHiddenLayers + 1; j2++) {
							h += weightMatrix[j1][j2]
									* this.perceptronMatrix[m][j2].getOutput();
						}
						this.perceptronMatrix[m + 1][j1].setH(h);
						this.perceptronMatrix[m + 1][j1]
								.setOutput(this.activationFunction.operate(h));
					}
				}
			}
		}
		/* There is no hidden layers. */
		else {
			for (int i = 0; i < this.nOutputs; i++) {
				double h = 0;
				for (int k = 0; k < this.nInputs + 1; k++) {
					h += this.weights[0][i][k]
							* this.perceptronMatrix[0][k].getOutput();
				}
				this.perceptronMatrix[1][i].setH(h);
				this.perceptronMatrix[1][i].setOutput(this.activationFunction
						.operate(h));
			}
		}
	}

	public void printWeights() {
		for (int m = 0; m < this.nHiddenLayers + 1; m++) {
			double[][] weightsAtLayer = this.weights[m];
			for (int i = 0; i < weightsAtLayer.length; i++) {
				double[] weightsAtPerceptron = weightsAtLayer[i];
				for (int j = 0; j < weightsAtPerceptron.length; j++) {
					System.out.println(String.format(
							"weight at m=%d, i=%d, j=%d: %g", m, i, j,
							weightsAtPerceptron[j]));
				}
			}
		}
	}

	private double randomWeight() {
		return Math.random() - 0.5;
	}

	private double[][] cloneWeigthMatrix(double[][] weightMatrix) {

		double[][] newMatrix = new double[weightMatrix.length][weightMatrix[0].length];
		for (int i = 0; i < weightMatrix.length; i++) {
			for (int j = 0; j < weightMatrix[i].length; j++) {
				newMatrix[i][j] = weightMatrix[i][j];
			}
		}
		return newMatrix;
	}

	private double[][][] cloneWeights(double[][][] weights) {
		double[][][] newArrayMatrix = new double[weights.length][][];
		for (int m = 0; m < weights.length; m++) {
			newArrayMatrix[m] = cloneWeigthMatrix(weights[m]);
		}
		return newArrayMatrix;
	}

	/**
	 * @param trainningSet
	 *            the trainning Set of Patterns
	 * @param errorRate
	 *            the error accepted between the real output and the Neural
	 *            Network Output
	 * @return the average of the well learned patterns over the quantity of all
	 *         patterns
	 */
	public double checkLearnedPatterns(List<Pattern> trainningSet,
			double errorRate) {

		double average = 0;

		for (Pattern currentPattern : trainningSet) {
			double[] inputCurrentPattern = currentPattern.getInput();
			double[] outputCurrentPattern = currentPattern.getOutput();
			double[] NNresult = this.testNeuralNetwork(inputCurrentPattern);

			
			boolean rejected = false;
			for (int i = 0; i < this.nOutputs; i++) {

				if (Math.abs(NNresult[i] - outputCurrentPattern[i]) > errorRate) {
					rejected = true;
				}
			}
			if(!rejected){
				average++;
			}

		}

		return average / trainningSet.size();

	}
}
