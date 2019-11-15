"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.MultinomialNB = void 0;

var _mlMatrix = _interopRequireDefault(require("ml-matrix"));

var _evaluation = require("./utils/evaluation");

var _methods = require("./utils/methods");

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

class MultinomialNB {
  constructor(model) {
    if (model) {
      this.conditionalProbability = _mlMatrix.default.checkMatrix(model.conditionalProbability);
      this.priorProbability = _mlMatrix.default.checkMatrix(model.priorProbability);
      this.classes = model.classes;
    }

    this.name = 'MultinomialNB';
  }

  fit(trainingSet, trainingLabels) {
    trainingSet = _mlMatrix.default.checkMatrix(trainingSet);

    if (trainingSet.rows !== trainingLabels.length) {
      throw new RangeError('the size of the training set and the training labels must be the same.');
    }

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
      this.conditionalProbability.setRow(i, new _mlMatrix.default([classValues.sum('column')]).add(1).div(divisor).apply(matrixLog));
    }

    this.classes = (0, _evaluation.getClassesList)(trainingLabels);
  }

  predict(dataset) {
    dataset = _mlMatrix.default.checkMatrix(dataset);
    var predictions = new Array(dataset.rows);

    for (var i = 0; i < dataset.rows; ++i) {
      var currentElement = dataset.getRowVector(i);
      predictions[i] = this.classes[new _mlMatrix.default([this.conditionalProbability.clone().mulRowVector(currentElement).sum('row')]).transpose().add(this.priorProbability).maxIndex()[0]];
    }

    return predictions;
  }

  predict_proba(dataset) {
    dataset = _mlMatrix.default.checkMatrix(dataset);
    var predictions = new Array(dataset.rows);

    for (var i = 0; i < dataset.rows; ++i) {
      var currentElement = dataset.getRowVector(i);
      predictions[i] = new _mlMatrix.default(this.conditionalProbability.clone().mulRowVector(currentElement).sum('row')).transpose().add(this.priorProbability);
    }

    return predictions;
  }

  save() {
    return {
      name: 'MultinomialNB',
      priorProbability: this.priorProbability,
      conditionalProbability: this.conditionalProbability,
      classes: this.classes
    };
  }

  static load(model) {
    if (model.name !== 'MultinomialNB') {
      throw new RangeError(`${model.name} is not a Multinomial Naive Bayes`);
    }

    return new MultinomialNB(model);
  }

}

exports.MultinomialNB = MultinomialNB;

function matrixLog(i, j) {
  this.set(i, j, Math.log10(this.get(i, j)));
}