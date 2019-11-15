"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.loadModelFromImportedData = loadModelFromImportedData;

var _index = require("./../index");

//  ----------------------- Inport Model ----------------------------

/**
 * @public
 * Function that returns model Instance from import model.
 * @param {Object} data - Model Data
 * @return {Object} - Model Instance
 */
function loadModelFromImportedData(data) {
  let model;
  if (data.name === 'NaiveBayes') model = new _index.NaiveBayes(data);
  if (data.name === 'MultinomialNB') model = new _index.MultinomialNB(data);
  if (data.name === 'GaussianNB') model = new _index.GaussianNB(data);
  if (data.name === 'BernoulliNB') model = new _index.BernoulliNB(data);
  return model;
}