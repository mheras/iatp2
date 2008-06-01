package ar.edu.itba.tp2.engine.pattern;

public class PatternListReport {

	private double maxError;

	private double minError;
	
	private double averageError;

	public double getAverageError() {
		return averageError;
	}

	public void setAverageError(double averageError) {
		this.averageError = averageError;
	}

	public double getMaxError() {
		return maxError;
	}

	public void setMaxError(double maxError) {
		this.maxError = maxError;
	}

	public double getMinError() {
		return minError;
	}

	public void setMinError(double minError) {
		this.minError = minError;
	}
}
