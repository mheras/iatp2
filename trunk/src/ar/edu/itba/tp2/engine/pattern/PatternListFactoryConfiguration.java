package ar.edu.itba.tp2.engine.pattern;

import ar.edu.itba.tp2.engine.function.Function;
import ar.edu.itba.tp2.engine.sigmoidfunction.SigmoidFunction;

public class PatternListFactoryConfiguration {

	private long quantity;
	private SigmoidFunction sigmoidFunction;
	private Function realFunction;
	

	public long getQuantity() {
		return quantity;
	}
	public void setQuantity(long quantity) {
		this.quantity = quantity;
	}
	public Function getRealFunction() {
		return realFunction;
	}
	public void setRealFunction(Function realFunction) {
		this.realFunction = realFunction;
	}
	public SigmoidFunction getSigmoidFunction() {
		return sigmoidFunction;
	}
	public void setSigmoidFunction(SigmoidFunction sigmoidFunction) {
		this.sigmoidFunction = sigmoidFunction;
	}
	
}
