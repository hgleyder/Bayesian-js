import { MultinomialNB, GaussianNB, BernoulliNB, NaiveBayes } from './../index';

//  ----------------------- Inport Model ----------------------------
/**
 * @public
 * Function that returns model Instance from import model.
 * @param {Object} data - Model Data
 * @return {Object} - Model Instance
 */
export function loadModelFromImportedData(data) {
	let model;
	if (data.name === 'NaiveBayes') model = new NaiveBayes(data);
	if (data.name === 'MultinomialNB') model = new MultinomialNB(data);
	if (data.name === 'GaussianNB') model = new GaussianNB(data);
	if (data.name === 'BernoulliNB') model = new BernoulliNB(data);
	return model;
}
