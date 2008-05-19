package ar.edu.itba.tp2.engine;

import java.util.Collection;

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
	private double[][][] weights;
	private Perceptron[][] perceptronMatrix;

	public BackPropagation(Integer nInputs, Integer nOutputs,
			Integer nHiddenLayers, Integer nNeuronsInHiddenLayers,
			SigmoidFunction activationFunction) {
		this.nInputs = nInputs;
		this.nOutputs = nOutputs;
		this.nHiddenLayers = nHiddenLayers;
		this.nNeuronsInHiddenLayers = nNeuronsInHiddenLayers;
		this.activationFunction = activationFunction;
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

	public void trainNeuralNetwork(Collection<Pattern> trainPatternSet) {

		for (Pattern currentPattern : trainPatternSet) {

			double[] inputs = currentPattern.getInput();
			/*
			 * Checks the length of the inputs corresponds to the length of the
			 * neural network.
			 */
			if (inputs.length != this.nInputs)
				throw new InvalidTrainPatternException();

			/*
			 * Stage 2: Puts the input in the neural network. The input layer is
			 * the 0 layer.
			 */
			for (int k = 0; k < this.nInputs; k++) {
				this.perceptronMatrix[0][k].setOutput(inputs[k]);
			}

			/* Stage 3: Propagates the signal forwards throught the network. */
			this.propagateInputSignal();

			/* Stage 4: Compute deltas for the output layer. */
			double[] deltasOutputLayer = this
					.computeDeltasOutputLayer(currentPattern.getOutput());

			/* Stage 5 and 6 still unimplemented. 
			 * See http://richardbowles.tripod.com/neural/backprop/backprop.htm 
			 */
		}

	}

	private double[] computeDeltasOutputLayer(double[] correctOutputs) {
		double[] result = new double[this.nOutputs];
		Perceptron[] NNoutputs = this.perceptronMatrix[this.nHiddenLayers + 1];
		for (int i = 0; i < this.nOutputs; i++) {
			result[i] = this.activationFunction.derivativeOperate(NNoutputs[i]
					.getH())
					* (correctOutputs[i] - NNoutputs[i].getOutput());
		}
		return result;
	}

	private void propagateInputSignal() {

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
	}
	
	private double randomWeight() {
		return Math.random() * 0.3;
	}
}
