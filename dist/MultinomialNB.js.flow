import Matrix from 'ml-matrix';
import { getClassesList } from './utils/evaluation';
import { getValuesByClasses } from './utils/methods';

export class MultinomialNB {
	constructor(model) {
		if (model) {
			this.conditionalProbability = Matrix.checkMatrix(
				model.conditionalProbability,
			);
			this.priorProbability = Matrix.checkMatrix(model.priorProbability);
			this.classes = model.classes;
		}
		this.name = 'MultinomialNB';
	}

	fit(trainingSet, trainingLabels) {
		trainingSet = Matrix.checkMatrix(trainingSet);

		if (trainingSet.rows !== trainingLabels.length) {
			throw new RangeError(
				'the size of the training set and the training labels must be the same.',
			);
		}

		var separateClass = getValuesByClasses(trainingSet, trainingLabels);
		this.priorProbability = new Matrix(separateClass.length, 1);

		for (var i = 0; i < separateClass.length; ++i) {
			this.priorProbability.set(i, 0, Math.log10(
				separateClass[i].rows / trainingSet.rows,
			));
		}


		var features = trainingSet.columns;
		this.conditionalProbability = new Matrix(
			separateClass.length,
			features,
		);
		for (i = 0; i < separateClass.length; ++i) {
			var classValues = Matrix.checkMatrix(separateClass[i]);
			var total = classValues.sum();
			var divisor = total + features;
			this.conditionalProbability.setRow(
				i,
				new Matrix([classValues.sum('column')]).add(1).div(divisor).apply(matrixLog),
			);
		}

		this.classes = getClassesList(trainingLabels);
	}

	predict(dataset) {
		dataset = Matrix.checkMatrix(dataset);
		var predictions = new Array(dataset.rows);
		for (var i = 0; i < dataset.rows; ++i) {
			var currentElement = dataset.getRowVector(i);
			predictions[i] = this.classes[new Matrix([this.conditionalProbability
				.clone()
				.mulRowVector(currentElement)
				.sum('row')])
				.transpose()
				.add(this.priorProbability)
				.maxIndex()[0]];
		}

		return predictions;
	}

	predict_proba(dataset) {
		dataset = Matrix.checkMatrix(dataset);
		var predictions = new Array(dataset.rows);
		for (var i = 0; i < dataset.rows; ++i) {
			var currentElement = dataset.getRowVector(i);
			predictions[i] = new Matrix(this.conditionalProbability
				.clone()
				.mulRowVector(currentElement)
				.sum('row'))
				.transpose()
				.add(this.priorProbability);
		}

		return predictions;
	}

	save() {
		return {
			name: 'MultinomialNB',
			priorProbability: this.priorProbability,
			conditionalProbability: this.conditionalProbability,
			classes: this.classes,
		};
	}

	static load(model) {
		if (model.name !== 'MultinomialNB') {
			throw new RangeError(
				`${model.name} is not a Multinomial Naive Bayes`,
			);
		}

		return new MultinomialNB(model);
	}
}

function matrixLog(i, j) {
	this.set(i, j, Math.log10(this.get(i,j)));
}
