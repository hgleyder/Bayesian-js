"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.NaiveBayes = void 0;

var _mlMatrix = _interopRequireDefault(require("ml-matrix"));

var _methods = require("./utils/methods");

var _evaluation = require("./utils/evaluation");

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

class NaiveBayes {
  constructor(model) {
    if (model) {
      this.probabilities = model.probabilities;
      this.classes = model.classes;
    }
  }

  fit(trainingSet, trainingLabels) {
    trainingSet = _mlMatrix.default.checkMatrix(trainingSet);

    if (trainingSet.rows !== trainingLabels.length) {
      throw new RangeError('the size of the training set and the training labels must be the same.');
    }

    const instancesByClass = (0, _methods.getValuesByClasses)(trainingSet, trainingLabels);
    const classList = (0, _evaluation.getClassesList)(trainingLabels);
    let Probabilities = {}; // Initialize probabilities for each attribute

    trainingSet[0].map((x, ind) => {
      Probabilities[ind] = {};
    }); // Initialize probabilities for each attribute value

    for (let index = 0; index < trainingSet[0].length; index++) {
      trainingSet.map(instance => {
        const aux = {};
        classList.map(c => {
          aux[c] = 0;
        });
        Probabilities[index][instance[index]] = aux;
      });
    } // Calculate occurrency for every attribute possible value per class


    Object.keys(Probabilities).map(attribute => {
      const values = Probabilities[attribute];
      Object.keys(values).map(value => {
        const AuxClasses = values[value];
        Object.keys(AuxClasses).map(classAux => {
          const auxClassList = classList.map(c => c.toString());
          const classIndex = auxClassList.indexOf(classAux);
          const relevant = instancesByClass[classIndex].filter(instance => instance[parseInt(attribute)] === value);
          Probabilities[attribute][value][classAux] = relevant.length; // Laplace +1 to every count

          Probabilities[attribute][value][classAux] += 1;
        });
      });
    }); // Calculate Probability for every attribute possible value per class

    Object.keys(Probabilities).map(attribute => {
      const values = Probabilities[attribute];
      Object.keys(values).map(value => {
        const AuxClasses = values[value];
        let total = 0;
        Object.keys(AuxClasses).map(classAux => {
          total = total + Probabilities[attribute][value][classAux];
        });
        Object.keys(AuxClasses).map(classAux => {
          Probabilities[attribute][value][classAux] = parseFloat(Probabilities[attribute][value][classAux] / total).toFixed(3);
        });
      });
    });
    this.probabilities = Probabilities;
    this.classes = (0, _evaluation.getClassesList)(trainingLabels);
  }

  predict(dataset) {
    dataset = _mlMatrix.default.checkMatrix(dataset);
    var predictions = new Array(dataset.rows);

    for (var pIndex = 0; pIndex < dataset.rows; ++pIndex) {
      let probabilitiesByClass = {};
      this.classes.map(c => {
        probabilitiesByClass[c] = 1;
      });

      for (var i = 0; i < dataset.rows; ++i) {
        var currentElement = dataset.getRowVector(i);
        const attributesLength = currentElement.length;
        this.classes.map(c => {
          for (let index = 0; index < attributesLength; index++) {
            const prop = this.probabilities[index.toString()][currentElement[0][index]][c.toString()];
            probabilitiesByClass[c] = probabilitiesByClass[c] * parseFloat(prop);
          }
        });
      }

      let maxProb = 0;
      let predClass = 0;
      Object.keys(probabilitiesByClass).map((c, ind) => {
        predClass = maxProb < probabilitiesByClass[c] ? ind : predClass;
        maxProb = maxProb < probabilitiesByClass[c] ? probabilitiesByClass[c] : maxProb;
      });
      predictions[pIndex] = predClass;
    }

    return predictions;
  }

  save() {
    return {
      name: 'NaiveBayes',
      probabilities: this.probabilities,
      classes: this.classes
    };
  }

  static load(model) {
    if (model.name !== 'NaiveBayes') {
      throw new RangeError(`${model.name} is not a Naive Bayes`);
    }

    return new NaiveBayes(model);
  }

}

exports.NaiveBayes = NaiveBayes;