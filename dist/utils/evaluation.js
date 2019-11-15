"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.crossValidationModel = crossValidationModel;
exports.getMetricsFromPredictions = getMetricsFromPredictions;
exports.crossValidationModelSpam = crossValidationModelSpam;
exports.getClassesList = getClassesList;
exports.getGeneralAccuracy = getGeneralAccuracy;
exports.getClassPrecision = getClassPrecision;
exports.getClassRecall = getClassRecall;
exports.calculateFMeasure = calculateFMeasure;
exports.getClassConfusionMatrix = getClassConfusionMatrix;

function crossValidationModel(classificationModel, X, y, folds = 10) {
  try {
    const instancesAmount = y.length;
    const instancesPerFold = Math.floor(instancesAmount / folds);
    const clasificationResults = [];
    let counter = 0;

    while (counter < folds) {
      try {
        let auxXtest = [];
        let auxYtest = [];
        var model = new classificationModel();
        var auxX = X.map(item => item);
        var auxY = y.map(item => item);
        auxXtest = auxX.splice(counter * instancesPerFold, instancesPerFold);
        auxYtest = auxY.splice(counter * instancesPerFold, instancesPerFold);
        var auxClassList = getClassesList(auxY);
        model.fit(auxX, auxY);
        var predictions = model.predict(auxXtest);
        predictions.map((prediction, i) => {
          clasificationResults.push({
            prediction: auxClassList[prediction].toString(),
            expected: auxYtest[i].toString()
          });
        });
      } catch (e) {
        console.log(e);
      }

      counter++;
    }

    const metrics = {};
    const classList = getClassesList(y);
    metrics['generalAccuracy'] = getGeneralAccuracy(clasificationResults);
    metrics['precisionByClasses'] = {};
    metrics['recallByClasses'] = {};
    metrics['fMeasureByClasses'] = {};
    metrics['confusionMatrixByClasses'] = {};
    classList.map(c => {
      metrics['precisionByClasses'][c.toString()] = getClassPrecision(clasificationResults, c);
      metrics['recallByClasses'][c.toString()] = getClassRecall(clasificationResults, c);
      metrics['fMeasureByClasses'][c.toString()] = calculateFMeasure(getClassPrecision(clasificationResults, c), getClassRecall(clasificationResults, c));
      metrics['confusionMatrixByClasses'][c.toString()] = getClassConfusionMatrix(clasificationResults, c, classList);
    });
    return metrics;
  } catch (e) {
    throw new Error('Incorrect type of data for this model');
  }
}

function getMetricsFromPredictions(y, yPredict) {
  try {
    const clasificationResults = [];
    yPredict.map((prediction, i) => {
      clasificationResults.push({
        prediction,
        expected: y[i][0]
      });
    });
    console.log(clasificationResults);
    const metrics = {};
    const classList = getClassesList(y);
    metrics['generalAccuracy'] = getGeneralAccuracy(clasificationResults);
    metrics['precisionByClasses'] = {};
    metrics['recallByClasses'] = {};
    metrics['fMeasureByClasses'] = {};
    metrics['confusionMatrixByClasses'] = {};
    classList.map(c => {
      metrics['precisionByClasses'][c.toString()] = getClassPrecision(clasificationResults, c);
      metrics['recallByClasses'][c.toString()] = getClassRecall(clasificationResults, c);
      metrics['fMeasureByClasses'][c.toString()] = calculateFMeasure(getClassPrecision(clasificationResults, c), getClassRecall(clasificationResults, c));
      metrics['confusionMatrixByClasses'][c.toString()] = getClassConfusionMatrix(clasificationResults, c, classList);
    });
    return metrics;
  } catch (e) {
    throw new Error('Incorrect type of data for this model');
  }
}

function crossValidationModelSpam(classificationModel, X, y, folds = 10) {
  try {
    const instancesAmount = y.length;
    const instancesPerFold = Math.floor(instancesAmount / folds);
    const clasificationResults = [];
    let counter = 0;

    while (counter < folds) {
      try {
        let auxXtest = [];
        let auxYtest = [];
        var model = new classificationModel();
        var auxX = X.map(item => item);
        var auxY = y.map(item => item);
        auxXtest = auxX.splice(counter * instancesPerFold, instancesPerFold);
        auxYtest = auxY.splice(counter * instancesPerFold, instancesPerFold);
        var auxClassList = getClassesList(auxY);
        model.fit(auxX, auxY);
        var predictions = model.predict(auxXtest);
        var predictionsProbs = model.predict_proba(auxXtest);
        const valsPreds = predictionsProbs.map(pred => pred.map(v => Math.pow(10, v)));
        const probs = valsPreds.map(p => {
          const total = p[0] + p[1];
          return [p[0] / total, p[1] / total];
        });
        predictions.map((prediction, i) => {
          clasificationResults.push({
            prediction: parseInt(prediction) === 1 ? probs[i][1] >= 0.7 ? auxClassList[1].toString() : auxClassList[0].toString() : auxClassList[prediction].toString(),
            expected: auxYtest[i].toString()
          });
        });
      } catch (e) {
        console.log(e);
      }

      counter++;
    }

    const metrics = {};
    const classList = getClassesList(y);
    metrics['generalAccuracy'] = getGeneralAccuracy(clasificationResults);
    metrics['precisionByClasses'] = {};
    metrics['recallByClasses'] = {};
    metrics['fMeasureByClasses'] = {};
    metrics['confusionMatrixByClasses'] = {};
    classList.map(c => {
      metrics['precisionByClasses'][c.toString()] = getClassPrecision(clasificationResults, c);
      metrics['recallByClasses'][c.toString()] = getClassRecall(clasificationResults, c);
      metrics['fMeasureByClasses'][c.toString()] = calculateFMeasure(getClassPrecision(clasificationResults, c), getClassRecall(clasificationResults, c));
      metrics['confusionMatrixByClasses'][c.toString()] = getClassConfusionMatrix(clasificationResults, c, classList);
    });
    return metrics;
  } catch (e) {
    throw new Error('Incorrect type of data for this model');
  }
}

function getClassesList(y) {
  var classesList = [];
  y.map(c => {
    if (!classesList.includes(c)) classesList.push(c);
  });
  return classesList;
} //  ----------------------- Metrics ----------------------------


function getGeneralAccuracy(predictions) {
  const predictionsLength = predictions.length;
  let correctlyInstantiatedCounter = 0;
  predictions.map(p => {
    if (p['prediction'] === p['expected']) correctlyInstantiatedCounter++;
  });
  return `${(parseFloat(correctlyInstantiatedCounter / predictionsLength) * 100).toFixed(4)}%`;
}

function getClassPrecision(predictions, currentClass) {
  const auxPredictions = predictions.filter(p => p.expected === currentClass.toString());
  let tp = 0;
  let fp = 0;
  auxPredictions.map(p => {
    if (p['prediction'] === p['expected']) tp++;
  });
  predictions.map(p => {
    if (p['prediction'] === currentClass.toString() && p['prediction'] !== p['expected']) fp++;
  });
  return parseFloat(tp / (tp + fp)).toFixed(4);
}

function getClassRecall(predictions, currentClass) {
  const auxPredictions = predictions.filter(p => p.expected === currentClass.toString());
  let tp = 0;
  let fn = 0;
  auxPredictions.map(p => {
    if (p['prediction'] === p['expected']) tp++;
  });
  predictions.map(p => {
    if (p['prediction'] !== currentClass.toString() && p['expected'] === currentClass.toString()) fn++;
  });
  return parseFloat(tp / (tp + fn)).toFixed(4);
}

function calculateFMeasure(precision, recall) {
  const p = parseFloat(precision);
  const r = parseFloat(recall);
  return (2 * (p * r / (p + r))).toFixed(4);
}

function getClassConfusionMatrix(predictions, currentClass, classList) {
  const auxPredictions = predictions.filter(p => p.expected === currentClass.toString());
  let aux = {};
  classList.map(c => {
    aux[c.toString()] = 0;
  });
  auxPredictions.map(prediction => {
    aux[prediction.prediction.toString()] += 1;
  });
  return aux;
}