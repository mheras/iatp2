package ar.edu.itba.tp2.engine.pattern;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;

import ar.edu.itba.tp2.engine.function.Function;
import ar.edu.itba.tp2.engine.sigmoidfunction.SigmoidFunction;
import ar.edu.itba.tp2.engine.sigmoidfunction.exp.SigmoidExponentialFunctionImpl;
import ar.edu.itba.tp2.engine.sigmoidfunction.tanh.SigmoidTanHFunctionImpl;

public class PatternListFactory {

	public static int MAX_X = 2;

	public static int MAX_Y = 2;

	public static int MIN_X = -2;

	public static int MIN_Y = -2;

	public static double DIV_FACTOR = 2;

	public static double MAX_OUTPUT = 0.5;

	public PatternListReport getPatternListReport(List<Pattern> patternList,
			PatternListFactoryConfiguration configuration){
		
			double maxError = Double.MIN_VALUE;
			double minError = Double.MAX_VALUE;
			double averageError = 0;
			PatternListReport report = new PatternListReport();
			
			for(Pattern currenPattern: patternList){
				double [] inputs = currenPattern.getInput();
				double [] outputs = currenPattern.getOutput();
				
				double [] realOutputs = configuration.getRealFunction().operate(inputs);
				if(Math.abs(realOutputs[0] - outputs[0]) > maxError){
					maxError = Math.abs(realOutputs[0] - outputs[0]);
				}
				
				if(Math.abs(realOutputs[0] - outputs[0]) < minError){
					minError = Math.abs(realOutputs[0] - outputs[0]);
				}
				averageError += Math.abs(realOutputs[0] - outputs[0]); 
			}
			report.setAverageError(averageError / patternList.size());
			report.setMaxError(maxError);
			report.setMinError(minError);
			return report;
		
	}

	public List<Pattern> getUniformMeshPatternList(
			PatternListFactoryConfiguration configuration) {

		List<Pattern> patternList = new ArrayList<Pattern>();

		SigmoidFunction sigmoidFunction = configuration.getSigmoidFunction();
		Function realFunction = configuration.getRealFunction();
		long quantity = configuration.getQuantity();

		if (sigmoidFunction instanceof SigmoidTanHFunctionImpl) {
			double step = (double) (MAX_X - MIN_X) / quantity;
			for (double t = MIN_X; t <= MAX_X;) {
				for (double r = MIN_Y; r <= MAX_Y;) {
					double[] currentInputs = new double[] { t, r };
					double[] currentOutputs = realFunction
							.operate(currentInputs);
					/* Mapping of the inputs. */
					for (int i = 0; i < currentInputs.length; i++) {
						currentInputs[i] = currentInputs[i] / DIV_FACTOR;
					}

					/* Mapping for the outputs. */
					for (int i = 0; i < currentOutputs.length; i++) {
						currentOutputs[i] = currentOutputs[i] / MAX_OUTPUT;
					}

					Pattern currentPattern = new Pattern(currentInputs,
							currentOutputs);
					patternList.add(currentPattern);
					r += step;
				}
				t += step;
			}

		} else if (sigmoidFunction instanceof SigmoidExponentialFunctionImpl) {
			double step = (MAX_X - MIN_X) / quantity;
			for (double t = MIN_X; t <= MAX_X;) {
				for (double r = MIN_Y; r <= MAX_Y;) {
					double[] currentInputs = new double[] { t, r };
					double[] currentOutputs = realFunction
							.operate(currentInputs);
					/* Mapping of the inputs. */
					for (int i = 0; i < currentInputs.length; i++) {
						currentInputs[i] = (currentInputs[i] + MAX_X)
								/ (DIV_FACTOR + MAX_X);
					}

					/* Mapping for the outputs. */
					for (int i = 0; i < currentOutputs.length; i++) {
						currentOutputs[i] = (currentOutputs[i] / MAX_OUTPUT);
					}

					Pattern currentPattern = new Pattern(currentInputs,
							currentOutputs);
					patternList.add(currentPattern);
					r += step;
				}
				t += step;
			}

		}

		return patternList;

	}

	public List<Pattern> getRandomMeshPatternList(
			PatternListFactoryConfiguration configuration) {
		List<Pattern> patternList = new ArrayList<Pattern>();

		SigmoidFunction sigmoidFunction = configuration.getSigmoidFunction();
		Function realFunction = configuration.getRealFunction();
		long quantity = (long) Math.pow(configuration.getQuantity() + 1, 2);

		if (sigmoidFunction instanceof SigmoidTanHFunctionImpl) {
			for (long i = 0; i < quantity; i++) {
				double randomx = Math.random() * DIV_FACTOR * DIV_FACTOR
						- MAX_X;
				double randomy = Math.random() * DIV_FACTOR * DIV_FACTOR
						- MAX_Y;
				double[] currentInputs = new double[] { randomx, randomy };
				double[] currentOutputs = realFunction.operate(currentInputs);
				/* Mapping of the inputs. */
				for (int j = 0; j < currentInputs.length; j++) {
					currentInputs[j] = currentInputs[j] / DIV_FACTOR;
				}

				/* Mapping for the outputs. */
				for (int j = 0; j < currentOutputs.length; j++) {
					currentOutputs[j] = currentOutputs[j] / MAX_OUTPUT;
				}

				Pattern currentPattern = new Pattern(currentInputs,
						currentOutputs);
				patternList.add(currentPattern);

			}
		} else if (sigmoidFunction instanceof SigmoidExponentialFunctionImpl) {
			for (long i = 0; i < quantity; i++) {
				double randomx = Math.random() * DIV_FACTOR * DIV_FACTOR
						- MAX_X;
				double randomy = Math.random() * DIV_FACTOR * DIV_FACTOR
						- MAX_Y;
				double[] currentInputs = new double[] { randomx, randomy };
				double[] currentOutputs = realFunction.operate(currentInputs);
				/* Mapping of the inputs. */
				for (int j = 0; j < currentInputs.length; j++) {
					currentInputs[j] = (currentInputs[j] + MAX_X)
							/ (DIV_FACTOR + MAX_X);
				}

				/* Mapping for the outputs. */
				for (int j = 0; j < currentOutputs.length; j++) {
					currentOutputs[j] = (currentOutputs[j] / MAX_OUTPUT);
				}

				Pattern currentPattern = new Pattern(currentInputs,
						currentOutputs);
				patternList.add(currentPattern);

			}
		}
		return patternList;

	}

	public List<Pattern> reMapList(
			PatternListFactoryConfiguration configuration,
			List<Pattern> mappedList) {
		List<Pattern> patternList = new ArrayList<Pattern>();

		SigmoidFunction sigmoidFunction = configuration.getSigmoidFunction();

		if (sigmoidFunction instanceof SigmoidTanHFunctionImpl) {
			for (Pattern currentPattern : mappedList) {
				double[] inputs = currentPattern.getInput();
				double[] outputs = currentPattern.getOutput();
				double[] newInputs = new double[inputs.length];
				double[] newOutputs = new double[outputs.length];
				/* Mapping of the inputs. */
				for (int j = 0; j < inputs.length; j++) {
					newInputs[j] = (inputs[j] * DIV_FACTOR);
				}

				/* Mapping for the outputs. */
				for (int j = 0; j < outputs.length; j++) {
					newOutputs[j] = (outputs[j] * MAX_OUTPUT);
				}
				Pattern newPattern = new Pattern(newInputs, newOutputs);
				patternList.add(newPattern);

			}
		} else if (sigmoidFunction instanceof SigmoidExponentialFunctionImpl) {
			for (Pattern currentPattern : mappedList) {
				double[] inputs = currentPattern.getInput();
				double[] outputs = currentPattern.getOutput();
				double[] newInputs = new double[inputs.length];
				double[] newOutputs = new double[outputs.length];
				/* Mapping of the inputs. */
				for (int j = 0; j < inputs.length; j++) {
					newInputs[j] = (inputs[j] * DIV_FACTOR * DIV_FACTOR)
							- MAX_X;
				}

				/* Mapping for the outputs. */
				for (int j = 0; j < outputs.length; j++) {
					newOutputs[j] = (outputs[j] - MAX_OUTPUT);
				}
				Pattern newPattern = new Pattern(newInputs, newOutputs);
				patternList.add(newPattern);

			}
		}
		return patternList;
	}

	public void saveToFile(List<Pattern> patternList, String fileName) {

		try {
			File file = new File(fileName);
			OutputStream fos = new FileOutputStream(file);
			OutputStreamWriter fw = new OutputStreamWriter(fos);

			fw.write("\n");
			for (Pattern currentPattern : patternList) {
				double[] inputs = currentPattern.getInput();
				double[] outputs = currentPattern.getOutput();
				for (int i = 0; i < inputs.length; i++) {
					fw.append(Double.toString(inputs[i]) + " ");
				}
				for (int i = 0; i < outputs.length; i++) {
					fw.append(Double.toString(outputs[i]));
				}
				fw.append("\n");

			}

			fw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

}
