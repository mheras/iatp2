package ar.edu.itba.tp2.engine.configuration;

import ar.edu.itba.tp2.engine.sigmoidfunction.SigmoidFunction;

public class BackPropagationConfiguration {

	private SigmoidFunction activationFunction = null;
	private Integer nInputs;
	private Integer nOutputs;
	private Integer nHiddenLayers;
	private Integer nNeuronsInHiddenLayers;
	private double learningRate;
	private long maxEpochs;
	private double momentum;
	private double adaptableEtaAlpha;
	private double adaptableEtaBeta;
	
	public SigmoidFunction getActivationFunction() {
		return activationFunction;
	}
	public void setActivationFunction(SigmoidFunction activationFunction) {
		this.activationFunction = activationFunction;
	}
	public double getLearningRate() {
		return learningRate;
	}
	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}
	public long getMaxEpochs() {
		return maxEpochs;
	}
	public void setMaxEpochs(long maxEpochs) {
		this.maxEpochs = maxEpochs;
	}
	public double getMomentum() {
		return momentum;
	}
	public void setMomentum(double momentum) {
		this.momentum = momentum;
	}
	public Integer getNHiddenLayers() {
		return nHiddenLayers;
	}
	public void setNHiddenLayers(Integer hiddenLayers) {
		nHiddenLayers = hiddenLayers;
	}
	public Integer getNInputs() {
		return nInputs;
	}
	public void setNInputs(Integer inputs) {
		nInputs = inputs;
	}
	public Integer getNNeuronsInHiddenLayers() {
		return nNeuronsInHiddenLayers;
	}
	public void setNNeuronsInHiddenLayers(Integer neuronsInHiddenLayers) {
		nNeuronsInHiddenLayers = neuronsInHiddenLayers;
	}
	public Integer getNOutputs() {
		return nOutputs;
	}
	public void setNOutputs(Integer outputs) {
		nOutputs = outputs;
	}
	public double getAdaptableEtaAlpha() {
		return adaptableEtaAlpha;
	}
	public void setAdaptableEtaAlpha(double adaptableEtaAlpha) {
		this.adaptableEtaAlpha = adaptableEtaAlpha;
	}
	public double getAdaptableEtaBeta() {
		return adaptableEtaBeta;
	}
	public void setAdaptableEtaBeta(double adaptableEtaBeta) {
		this.adaptableEtaBeta = adaptableEtaBeta;
	}
	
}
