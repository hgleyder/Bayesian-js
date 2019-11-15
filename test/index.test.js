import { MultinomialNB, GaussianNB, BernoulliNB, NaiveBayes } from "../src";

test("MultinomialNB", () => {
  const model = new MultinomialNB();
  model.fit(
    [
      [2, 1, 0, 0, 0, 0],
      [2, 0, 1, 0, 0, 0],
      [1, 0, 0, 1, 0, 0],
      [1, 0, 0, 0, 1, 1]
    ],
    ["yes", "yes", "yes", "no"]
  );
  expect(model.predict([[3, 0, 0, 0, 1, 1]])).toEqual(["yes"]);
});

test("GaussianNB", () => {
  const model = new GaussianNB();
  model.fit(
    [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
    [1, 1, 1, 2, 2, 2]
  );
  expect(model.predict([[-0.8, -1]])).toEqual([1]);
});

test("BernoulliNB", () => {
  const model = new BernoulliNB();
  model.fit(
    [
      [1, 1, 0, 0, 0, 0],
      [1, 0, 1, 0, 0, 0],
      [1, 0, 0, 1, 0, 0],
      [1, 0, 0, 0, 1, 1]
    ],
    ["yes", "yes", "yes", "no"]
  );
  expect(model.predict([[1, 0, 1, 0, 1, 1]])).toEqual(["no"]);
});

test("NaiveBayes", () => {
  const model = new NaiveBayes();
  model.fit([[1, 1], [1, 2], [2, 2], [2, 3]], ["no", "no", "yes", "yes"]);
  expect(model.predict([[1, 3]])).toEqual(["no"]);
});
