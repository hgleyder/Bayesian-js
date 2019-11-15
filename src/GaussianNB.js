import Matrix from 'ml-matrix';
import Stat from 'ml-stat';

import { getValuesByClasses } from './utils/methods';
import { getClassesList } from './utils/evaluation';

export class GaussianNB {
	constructor(reload, model) {
		if (reload) {
			this.means = model.means;
			this.jointProbabilities = model.jointProbabilities;
			this.classes = model.classes;
		}
		this.name = 'GaussianNB';
	}

	fit(trainingSet, trainingLabels) {
		trainingSet = Matrix.checkMatrix(trainingSet);

		if (trainingSet.rows !== trainingLabels.length) {
			throw new RangeError(
				'the size of the training set and the training labels must be the same.',
			);
		}

		var separatedClasses = getValuesByClasses(trainingSet, trainingLabels);
		var jointProbabilities = new Array(separatedClasses.length);

		this.means = new Array(separatedClasses.length);
		for (var i = 0; i < separatedClasses.length; ++i) {
			var means = Stat.matrix.mean(separatedClasses[i].to2DArray());
			var std = calculateStandardDesviations(means, separatedClasses[i]);
			var logPriorProbability = Math.log10(
				separatedClasses[i].rows / trainingSet.rows,
			);
			jointProbabilities[i] = new Array(means.length + 1);

			jointProbabilities[i][0] = logPriorProbability;
			for (var j = 1; j < means.length + 1; ++j) {
				var currentStd = std[j - 1];
				jointProbabilities[i][j] = [
					1 / (Math.sqrt(2 * Math.PI) * currentStd),
					-2 * currentStd * currentStd,
				];
			}
			this.means[i] = means;
		}

		this.jointProbabilities = jointProbabilities;

		this.classes = getClassesList(trainingLabels);
	}

	predict(dataset) {
		if (dataset[0].length === this.jointProbabilities[0].length) {
			throw new RangeError(
				'the dataset must have the same features as the training set',
			);
		}

		var predictions = new Array(dataset.length);

		for (var i = 0; i < predictions.length; ++i) {
			predictions[i] = this.classes[getCurrentClass(
				dataset[i],
				this.means,
				this.jointProbabilities,
			)];
		}

		return predictions;
	}

	save() {
		return {
			modelName: 'GaussianNB',
			means: this.means,
			jointProbabilities: this.jointProbabilities,
			classes: this.classes,
		};
	}

	static load(model) {
		if (model.modelName !== 'GaussianNB') {
			throw new RangeError(
				'The current model is not a Multinomial Naive Bayes, current model:',
				model.name,
			);
		}

		return new GaussianNB(true, model);
	}
}

function getCurrentClass(currentCase, mean, classes) {
	var maxProbability = 0;
	var predictedClass = -1;

	// going through all precalculated values for the classes
	for (var i = 0; i < classes.length; ++i) {
		// intitialize current as the PriorProbability
		var currentProbability = classes[i][0];
		for (var j = 1; j < classes[0][1].length + 1; ++j) {
			currentProbability += calculateLogProbability(
				currentCase[j - 1],
				mean[i][j - 1],
				classes[i][j][0],
				classes[i][j][1],
			);
		}

		// get probability value
		currentProbability = Math.pow(10, currentProbability);

		if (currentProbability > maxProbability) {
			maxProbability = currentProbability;
			predictedClass = i;
		}
	}

	return predictedClass;
}

function calculateLogProbability(value, mean, C1, C2) {
	value = value - mean;
	return Math.log10(C1 * Math.exp(value * value / C2));
}

function calculateStandardDesviations(means, values) {
	let std = [];
	let value;
	for (let i = 0; i < means.length; i++) {
		value = 0;
		for (let j = 0; j < values.rows; j++) {
			value += Math.pow(Math.abs(values.get(j, i) - means[i]), 2);
		}
		value = value / values.rows;
		std.push(value);
	}
	return std;
}
