package ar.edu.itba.tp2.engine.pattern;

public class Pattern {

	private double [] input;
	private double [] output;
	
	public Pattern(double [] input, double [] output){
		this.input = input;
		this.output = output;
	}

	public double[] getInput() {
		return input;
	}

	public void setInput(double[] input) {
		this.input = input;
	}

	public double[] getOutput() {
		return output;
	}

	public void setOutput(double[] output) {
		this.output = output;
	}
	

}
