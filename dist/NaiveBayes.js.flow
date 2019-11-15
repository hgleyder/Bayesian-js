import Matrix from 'ml-matrix';

import { getValuesByClasses } from './utils/methods';
import { getClassesList } from './utils/evaluation';

export class NaiveBayes {
	constructor(model) {
		if (model) {
			this.probabilities = model.probabilities;
			this.classes = model.classes;
		}
		this.name = 'NaiveBayes';
	}

	fit(trainingSet, trainingLabels) {
		trainingSet = Matrix.checkMatrix(trainingSet);

		if (trainingSet.rows !== trainingLabels.length) {
			throw new RangeError(
				'the size of the training set and the training labels must be the same.',
			);
		}

		const instancesByClass = getValuesByClasses(
			trainingSet,
			trainingLabels,
		);

		const classList = getClassesList(trainingLabels);
		let Probabilities = {};

		// Initialize probabilities for each attribute
		trainingSet.to2DArray()[0].map((x, ind) => {
			Probabilities[ind] = {};
		});

	
		// Initialize probabilities for each attribute value
		for (let index = 0; index < trainingSet.to2DArray()[0].length; index++) {
			trainingSet.to2DArray().map((instance) => {
				const aux = {};
				classList.map((c) => {
					aux[c] = 0;
				});
				Probabilities[index][instance[index]] = aux;
			});
		}

		// Calculate occurrency for every attribute possible value per class
		Object.keys(Probabilities).map((attribute) => {
			const values = Probabilities[attribute];
			Object.keys(values).map((value) => {
				const AuxClasses = values[value];
				Object.keys(AuxClasses).map((classAux) => {
					const auxClassList = classList.map((c) => c.toString());
					const classIndex = auxClassList.indexOf(classAux);
					const relevant = instancesByClass[classIndex].to2DArray().filter(
						(instance) => instance[parseInt(attribute)] === parseInt(value),
					);
					Probabilities[attribute][value][classAux] = relevant.length;
					// Laplace +1 to every count
					Probabilities[attribute][value][classAux] += 1;
				});
			});
		});


		// Calculate Probability for every attribute possible value per class
		Object.keys(Probabilities).map((attribute) => {
			const values = Probabilities[attribute];
			Object.keys(values).map((value) => {
				const AuxClasses = values[value];
				let total = 0;
				Object.keys(AuxClasses).map((classAux) => {
					total = total + Probabilities[attribute][value][classAux];
				});
				Object.keys(AuxClasses).map((classAux) => {
					Probabilities[attribute][value][classAux] = parseFloat(
						Probabilities[attribute][value][classAux] / total,
					).toFixed(3);
				});
			});
		});

		this.probabilities = Probabilities;
		this.classes = getClassesList(trainingLabels);
	}

	predict(dataset) {
		dataset = Matrix.checkMatrix(dataset);
		var predictions = new Array(dataset.rows);
		for (var pIndex = 0; pIndex < dataset.rows; ++pIndex) {
			let probabilitiesByClass = {};
			this.classes.map((c) => {
				probabilitiesByClass[c] = 1;
			});
			for (var i = 0; i < dataset.rows; ++i) {
				var currentElement = dataset.getRowVector(i);
				const attributesLength = currentElement.length;
				this.classes.map((c) => {
					for (let index = 0; index < attributesLength; index++) {
						const prop = this.probabilities[index.toString()][
							currentElement[0][index]
						][c.toString()];
						probabilitiesByClass[c] =
							probabilitiesByClass[c] * parseFloat(prop);
					}
				});
			}
			let maxProb = 0;
			let predClass = 0;
			Object.keys(probabilitiesByClass).map((c, ind) => {
				predClass = maxProb < probabilitiesByClass[c] ? ind : predClass;
				maxProb =
					maxProb < probabilitiesByClass[c]
						? probabilitiesByClass[c]
						: maxProb;
			});

			predictions[pIndex] = this.classes[predClass];
		}

		return predictions;
	}

	save() {
		return {
			name: 'NaiveBayes',
			probabilities: this.probabilities,
			classes: this.classes,
		};
	}

	static load(model) {
		if (model.name !== 'NaiveBayes') {
			throw new RangeError(`${model.name} is not a Naive Bayes`);
		}

		return new NaiveBayes(model);
	}
}
