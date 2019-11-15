import Matrix from 'ml-matrix';

import { getValuesByClasses } from './utils/methods';
import { getClassesList } from './utils/evaluation';

export class BernoulliNB {
	constructor(model) {
		if (model) {
			this.conditionalProbability = Matrix.checkMatrix(
				model.conditionalProbability,
			);
			this.priorProbability = Matrix.checkMatrix(model.priorProbability);
			this.classes = model.classes;
		}
		this.name = 'BernoulliNB';
	}

	fit(trainingSet, trainingLabels) {
		trainingSet = Matrix.checkMatrix(trainingSet);

		if (trainingSet.rows !== trainingLabels.length) {
			throw new RangeError(
				'the size of the training set and the training labels must be the same.',
			);
		}

		// inspect if there is at least an string attribute
		const stringAttr = trainingSet.to2DArray()[0].find(
			(attr) => typeof attr === 'string' || attr instanceof String,
		);

		if (stringAttr) {
			throw new RangeError('the attributes should be numeric');
		}

		// converting values to 0 and 1
		let auxTraniningSet = trainingSet.to2DArray();
		auxTraniningSet.map((instance, ind) => {
			auxTraniningSet[ind].map((attr, ind2) => {
				auxTraniningSet[ind][ind2] = attr > 0 ? 1 : 0;
			});
		});


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
				new Matrix([classValues.sum('column')]).add(2).div(divisor),
			);
		}

		this.classes = getClassesList(trainingLabels);
	}

	predict(dataset) {
		dataset = Matrix.checkMatrix(dataset);
		let auxDataset = dataset.to2DArray();
		auxDataset.map((instance, ind) => {
			auxDataset[ind].map((attr, ind2) => {
				auxDataset[ind][ind2] = attr > 0 ? 1 : 0;
			});
		});
		auxDataset = new Matrix(auxDataset)
		var predictions = new Array(auxDataset.rows);
		for (var i = 0; i < auxDataset.rows; ++i) {
			var currentElement = auxDataset.getRowVector(i);

			let auxProb = [];
			for (let i = 0; i < this.conditionalProbability.rows; i++) {
				auxProb.push([]);
				for (
					let j = 0;
					j < this.conditionalProbability.columns;
					j++
				) {
					auxProb[i].push(1 - this.conditionalProbability.get(i,j));
				}
			}
			auxProb = new Matrix(auxProb);

			var inverseCurrent = [];
			for (let index = 0; index < currentElement.columns; index++) {
				inverseCurrent.push(
					Math.abs(parseInt(currentElement.get(0,index)) - 1),
				);
			}
			inverseCurrent = new Matrix([ inverseCurrent ]);

			predictions[i] = this.classes[new Matrix([this.conditionalProbability
				.clone()
				.apply(matrixLog)
				.mulRowVector(currentElement)
				.add(auxProb.apply(matrixLog)
				.mulRowVector(inverseCurrent))
				.sum('row')])
				.transpose()
				.add(this.priorProbability)
				.maxIndex()[0]];
		}


		return predictions;
	}

	save() {
		return {
			name: 'BernoulliNB',
			priorProbability: this.priorProbability,
			conditionalProbability: this.conditionalProbability,
			classes: this.classes,
		};
	}

	static load(model) {
		if (model.name !== 'BernoulliNB') {
			throw new RangeError(
				`${model.name} is not a Bernoulli Naive Bayes`,
			);
		}

		return new BernoulliNB(model);
	}
}

function matrixLog(i, j) {
	this.set(i, j, Math.log10(this.get(i,j)));
}
