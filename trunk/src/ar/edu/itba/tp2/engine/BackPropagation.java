package ar.edu.itba.tp2.engine;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import ar.edu.itba.tp2.engine.exception.InvalidInputException;
import ar.edu.itba.tp2.engine.exception.InvalidTrainPatternException;
import ar.edu.itba.tp2.engine.pattern.Pattern;
import ar.edu.itba.tp2.engine.sigmoidfunction.SigmoidFunction;

/**
 * @author Jorge Goldman
 * 
 * This class is a BackPropagtion Learning Neural Network.
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
	private long maxEpochs;
	private Perceptron[][] perceptronMatrix;

	public BackPropagation(Integer nInputs, Integer nOutputs,
			Integer nHiddenLayers, Integer nNeuronsInHiddenLayers,
			double learningRate, long maxEpochs,
			SigmoidFunction activationFunction) {
		this.nInputs = nInputs;
		this.nOutputs = nOutputs;
		this.nHiddenLayers = nHiddenLayers;
		this.nNeuronsInHiddenLayers = nNeuronsInHiddenLayers;
		this.activationFunction = activationFunction;
		this.learningRate = learningRate;
		this.maxEpochs = maxEpochs;
		weights = new double[nHiddenLayers + 1][][];
		perceptronMatrix = new Perceptron[nHiddenLayers + 2][];

		/* Stage 1: Initialize the weights. */
		this.initializePerceptrons();
		this.initializeWeightMatrix();
	}

	private void initializePerceptrons() {

		/*
		 * Iterates over the layers of the neural network, the layer 0 is the
		 * input layer. The N - 1 layer is the output layer.
		 */
		for (int i = 0; i < this.nHiddenLayers + 2; i++) {

			/* The input layer. */
			if (i == 0) {
				this.perceptronMatrix[i] = new Perceptron[this.nInputs + 1];
				for (int j = 0; j < this.nInputs + 1; j++) {
					this.perceptronMatrix[i][j] = new Perceptron();
				}
				/* The BIAS Input. */
				this.perceptronMatrix[i][this.nInputs].setOutput(-1);
			}
			/* The output layer. */
			else if (i == (this.nHiddenLayers + 1)) {
				this.perceptronMatrix[i] = new Perceptron[this.nOutputs];

				for (int j = 0; j < this.nOutputs; j++) {
					this.perceptronMatrix[i][j] = new Perceptron();
				}
			}
			/* The Intermediate Hidden Layers. */
			else {
				this.perceptronMatrix[i] = new Perceptron[this.nNeuronsInHiddenLayers + 1];
				for (int j = 0; j < this.nNeuronsInHiddenLayers + 1; j++) {
					this.perceptronMatrix[i][j] = new Perceptron();
				}
				/* The BIAS Input in the Hidden Layer i. */
				this.perceptronMatrix[i][this.nNeuronsInHiddenLayers]
						.setOutput(-1);
			}

		}

	}

	/**
	 * This method initialize the array of weight matrices.
	 */
	private void initializeWeightMatrix() {

		if (this.nHiddenLayers == 0) {
			this.weights[0] = new double[this.nInputs + 1][this.nOutputs];
			for (int i = 0; i < this.nInputs + 1; i++) {
				for (int j = 0; j < this.nOutputs; j++) {
					this.weights[0][i][j] = randomWeight();
				}
			}
			return;
		}

		for (int i = 0; i < this.nHiddenLayers + 1; i++) {
			/* Weights between the input layer and the first hidden layer. */
			if (i == 0) {
				this.weights[i] = new double[this.nInputs + 1][this.nNeuronsInHiddenLayers];
				for (int j = 0; j < this.nInputs + 1; j++) {
					for (int k = 0; k < this.nNeuronsInHiddenLayers; k++) {
						this.weights[i][j][k] = randomWeight();
					}
				}
			}
			/* Weights between the last hidden layer and the output. */
			else if (i == this.nHiddenLayers) {
				this.weights[i] = new double[this.nNeuronsInHiddenLayers + 1][this.nOutputs];
				for (int j = 0; j < this.nNeuronsInHiddenLayers + 1; j++) {
					for (int k = 0; k < this.nOutputs; k++) {
						this.weights[i][j][k] = randomWeight();
					}
				}
			}
			/* Weights between hidden layers. */
			else {
				this.weights[i] = new double[this.nNeuronsInHiddenLayers + 1][this.nNeuronsInHiddenLayers];
				for (int j = 0; j < this.nNeuronsInHiddenLayers + 1; j++) {
					for (int k = 0; k < this.nNeuronsInHiddenLayers; k++) {
						this.weights[i][j][k] = randomWeight();
					}
				}
			}

		}
	}

	/**
	 * @see http://richardbowles.tripod.com/neural/backprop/backprop.htm
	 */
	public void trainNeuralNetwork(Collection<Pattern> trainPatternSet) {

		for (long currentEpoch = 0; currentEpoch < this.maxEpochs; currentEpoch++) {
//			List <Pattern> unsortedPatternList = this.unsortSet();
			
			for (Pattern currentPattern : trainPatternSet) {

				double[] inputs = currentPattern.getInput();
				double[][] deltas = new double[this.nHiddenLayers + 2][];

				/*
				 * Checks the length of the inputs corresponds to the length of
				 * the neural network.
				 */
				if (inputs.length != this.nInputs)
					throw new InvalidTrainPatternException();

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

				/*
				 * Stage 5: Compute deltas for the preceding layers by
				 * propagating the errors backwards.
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
						if (m + 1 == this.nHiddenLayers + 1) {
							upperLayerCount = this.perceptronMatrix[m + 1].length;
						} else {
							upperLayerCount = this.perceptronMatrix[m + 1].length - 1;
						}

						for (int j = 0; j < upperLayerCount; j++) {
							delta += this.weights[m][i][j] * deltas[m + 1][j];
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

				if (this.nHiddenLayers != 0) {
					for (int m = 0; m < this.nHiddenLayers + 1; m++) {
						/* Weigths between the input and first hidden layer. */
						if (m == 0) {
							for (int i = 0; i < this.nInputs + 1; i++) {
								for (int j = 0; j < this.nHiddenLayers; j++) {
									this.weights[m][i][j] += this.learningRate
											* deltas[m + 1][j]
											* this.perceptronMatrix[m][i]
													.getOutput();
								}
							}
						} else if (m == this.nHiddenLayers) {
							for (int i = 0; i < this.nHiddenLayers + 1; i++) {
								for (int j = 0; j < this.nOutputs; j++) {
									this.weights[m][i][j] += this.learningRate
											* deltas[m + 1][j]
											* this.perceptronMatrix[m][i]
													.getOutput();
								}
							}
						} else {
							for (int i = 0; i < this.nHiddenLayers + 1; i++) {
								for (int j = 0; j < this.nHiddenLayers; j++) {
									this.weights[m][i][j] += this.learningRate
											* deltas[m + 1][j]
											* this.perceptronMatrix[m][i]
													.getOutput();
								}
							}
						}
					}
				} else {
					for (int i = 0; i < this.nInputs + 1; i++) {
						for (int j = 0; j < this.nOutputs; j++) {
							this.weights[0][i][j] += this.learningRate
									* deltas[1][j]
									* this.perceptronMatrix[0][i].getOutput();
						}
					}
				}
			}
			
//			System.out.println("**********************");
//			System.out.println(String.format("Epoch %d", currentEpoch));
//			this.printWeights();
		}
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
			result[j] = this.perceptronMatrix[this.nHiddenLayers+1][j]
					.getOutput();
		}
		return result;

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
		System.out.println(sum);
		return result;
	}

	private void propagateInputSignal() {

		if (this.nHiddenLayers != 0) {
			for (int i = 0; i < this.nHiddenLayers + 1; i++) {
				double[][] weightMatrix = this.weights[i];
				/* Between Inputs and Hidden Layers. Plus the BIAS. */
				if (i == 0) {

					for (int j = 0; j < this.nNeuronsInHiddenLayers; j++) {
						double h = 0;
						for (int k = 0; k < this.nInputs + 1; k++) {
							h += weightMatrix[k][j]
									* this.perceptronMatrix[i][k].getOutput();
						}
						this.perceptronMatrix[i + 1][j].setH(h);
						this.perceptronMatrix[i + 1][j]
								.setOutput(this.activationFunction.operate(h));
					}
				}
				/* Between Hidden Layer and Outputs. Plus the BIAS. */
				else if (i == this.nHiddenLayers) {

					for (int j = 0; j < this.nOutputs; j++) {
						double h = 0;
						for (int k = 0; k < this.nNeuronsInHiddenLayers + 1; k++) {
							h += weightMatrix[k][j]
									* this.perceptronMatrix[i][k].getOutput();
						}
						this.perceptronMatrix[i + 1][j].setH(h);
						this.perceptronMatrix[i + 1][j]
								.setOutput(this.activationFunction.operate(h));
					}

				} else {

					for (int j = 0; j < this.nNeuronsInHiddenLayers; j++) {

						double h = 0;
						for (int k = 0; k < this.nNeuronsInHiddenLayers + 1; k++) {
							h += weightMatrix[k][j]
									* this.perceptronMatrix[i][k].getOutput();
						}
						this.perceptronMatrix[i + 1][j].setH(h);
						this.perceptronMatrix[i + 1][j]
								.setOutput(this.activationFunction.operate(h));
					}
				}
			}
		} else {
			for (int j = 0; j < this.nOutputs; j++) {
				double h = 0;
				for (int k = 0; k < this.nInputs + 1; k++) {
					h += this.weights[0][k][j]
							* this.perceptronMatrix[0][k].getOutput();
				}
				this.perceptronMatrix[1][j].setH(h);
				this.perceptronMatrix[1][j].setOutput(this.activationFunction
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
		return Math.random() * 0.3;
	}
}
