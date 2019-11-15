"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.getValuesByClasses = getValuesByClasses;

var _mlMatrix = _interopRequireDefault(require("ml-matrix"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function getValuesByClasses(instances, labels) {
  var features = instances.columns;
  var totalPerClasses = {};

  for (var i = 0; i < labels.length; i++) {
    if (totalPerClasses[labels[i]] === undefined) {
      totalPerClasses[labels[i]] = 0;
    }

    totalPerClasses[labels[i]]++;
  }

  const classes = Object.keys(totalPerClasses).length;
  var separatedClasses = new Array(classes);
  var currentIndex = new Array(classes);

  for (i = 0; i < classes; i++) {
    const totalsKeys = Object.keys(totalPerClasses);
    separatedClasses[i] = new _mlMatrix.default(totalPerClasses[totalsKeys[i]], features);
    currentIndex[i] = 0;
  }

  const classesList = Object.keys(totalPerClasses);
  const aux = new Array(classesList.length);

  for (let index = 0; index < classesList.length; index++) {
    aux[index] = [];
  }

  for (i = 0; i < instances.rows; i++) {
    const auxIndex = classesList.indexOf(labels[i].toString());
    aux[auxIndex].push(instances.to2DArray()[i]);
    separatedClasses[auxIndex].set([currentIndex[auxIndex]], instances.to2DArray()[i]);
    currentIndex[auxIndex]++;
  }

  for (let index = 0; index < classesList.length; index++) {
    aux[index] = new _mlMatrix.default(aux[index]);
  }

  return aux;
}