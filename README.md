# BayesianJS
### Naive Bayes Machine Learning algorithms implemented on JS

## USAGE
```javascript
import { MultinomialNB, GaussianNB, BernoulliNB, NaiveBayes } from "bayesian-js";

...
...

//MultinomialNB
  const model = new MultinomialNB();
  model.fit([[2, 1, 0,0,0,0], [2,0,1,0,0,0], [1, 0,0, 1, 0, 0], [1, 0,0, 0,1,1]], ['yes', 'yes', 'yes', 'no']);
  console.log(model.predict([[3,0,0,0,1,1]]))
});

...

//GaussianNB NB
  const model = new GaussianNB();
  model.fit([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], [1, 1, 1, 2, 2, 2]);
  console.log(model.predict([[-0.8, -1]]))
});


...

//BernoulliNB
  const model = new BernoulliNB();
  model.fit([[1, 1, 0,0,0,0], [1,0,1,0,0,0], [1, 0,0, 1, 0, 0], [1, 0,0, 0,1,1]], ['yes', 'yes', 'yes', 'no']);
  console.log(model.predict([[1,0,1,0,1,1]]))
});

...

// Naive Bayes
test("NaiveBayes", () => {
  const model = new NaiveBayes();
  model.fit([[1, 1], [1, 2], [2, 2], [2, 3]], ['no', 'no', 'yes', 'yes']);
  console.log(model.predict([[1, 3]]))
});
```

## API

## License

MIT © [Gleyder Hernandez](https://github.com/hgleyder)
