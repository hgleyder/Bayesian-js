"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.BernoulliNB = void 0;

var _mlMatrix = _interopRequireDefault(require("ml-matrix"));

var _methods = require("./utils/methods");

var _evaluation = require("./utils/evaluation");

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

class BernoulliNB {
  constructor(model) {
    if (model) {
      this.conditionalProbability = _mlMatrix.default.checkMatrix(model.conditionalProbability);
      this.priorProbability = _mlMatrix.default.checkMatrix(model.priorProbability);
      this.classes = model.classes;
    }

    this.name = 'BernoulliNB';
  }

  fit(trainingSet, trainingLabels) {
    trainingSet = _mlMatrix.default.checkMatrix(trainingSet);

    if (trainingSet.rows !== trainingLabels.length) {
      throw new RangeError('the size of the training set and the training labels must be the same.');
    } // inspect if there is at least an string attribute


    const stringAttr = trainingSet.to2DArray()[0].find(attr => typeof attr === 'string' || attr instanceof String);

    if (stringAttr) {
      throw new RangeError('the attributes should be numeric');
    } // converting values to 0 and 1


    let auxTraniningSet = trainingSet.to2DArray();
    auxTraniningSet.map((instance, ind) => {
      auxTraniningSet[ind].map((attr, ind2) => {
        auxTraniningSet[ind][ind2] = attr > 0 ? 1 : 0;
      });
    });
    var separateClass = (0, _methods.getValuesByClasses)(trainingSet, trainingLabels);
    this.priorProbability = new _mlMatrix.default(separateClass.length, 1);

    for (var i = 0; i < separateClass.length; ++i) {
      this.priorProbability.set(i, 0, Math.log10(separateClass[i].rows / trainingSet.rows));
    }

    var features = trainingSet.columns;
    this.conditionalProbability = new _mlMatrix.default(separateClass.length, features);

    for (i = 0; i < separateClass.length; ++i) {
      var classValues = _mlMatrix.default.checkMatrix(separateClass[i]);

      var total = classValues.sum();
      var divisor = total + features;
      this.conditionalProbability.setRow(i, new _mlMatrix.default([classValues.sum('column')]).add(2).div(divisor));
    }

    this.classes = (0, _evaluation.getClassesList)(trainingLabels);
  }

  predict(dataset) {
    dataset = _mlMatrix.default.checkMatrix(dataset);
    let auxDataset = dataset.to2DArray();
    auxDataset.map((instance, ind) => {
      auxDataset[ind].map((attr, ind2) => {
        auxDataset[ind][ind2] = attr > 0 ? 1 : 0;
      });
    });
    auxDataset = new _mlMatrix.default(auxDataset);
    var predictions = new Array(auxDataset.rows);

    for (var i = 0; i < auxDataset.rows; ++i) {
      var currentElement = auxDataset.getRowVector(i);
      let auxProb = [];

      for (let i = 0; i < this.conditionalProbability.rows; i++) {
        auxProb.push([]);

        for (let j = 0; j < this.conditionalProbability.columns; j++) {
          auxProb[i].push(1 - this.conditionalProbability.get(i, j));
        }
      }

      auxProb = new _mlMatrix.default(auxProb);
      var inverseCurrent = [];

      for (let index = 0; index < currentElement.columns; index++) {
        inverseCurrent.push(Math.abs(parseInt(currentElement.get(0, index)) - 1));
      }

      inverseCurrent = new _mlMatrix.default([inverseCurrent]);
      predictions[i] = this.classes[new _mlMatrix.default([this.conditionalProbability.clone().apply(matrixLog).mulRowVector(currentElement).add(auxProb.apply(matrixLog).mulRowVector(inverseCurrent)).sum('row')]).transpose().add(this.priorProbability).maxIndex()[0]];
    }

    return predictions;
  }

  save() {
    return {
      name: 'BernoulliNB',
      priorProbability: this.priorProbability,
      conditionalProbability: this.conditionalProbability,
      classes: this.classes
    };
  }

  static load(model) {
    if (model.name !== 'BernoulliNB') {
      throw new RangeError(`${model.name} is not a Bernoulli Naive Bayes`);
    }

    return new BernoulliNB(model);
  }

}

exports.BernoulliNB = BernoulliNB;

function matrixLog(i, j) {
  this.set(i, j, Math.log10(this.get(i, j)));
}