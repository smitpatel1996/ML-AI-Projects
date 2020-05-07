# Common ML Models
## Linear Models
* Linear Regression
    * Training Method: **Normal Equation** -- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    * Training Method: **Gradient Descent** -- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html
* Polynomial Regression
    * Feature Addition: **Polynomial Features** -- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html (after that, use *Linear Regression()* to fit non-linear model)
* Regression With Regularization
    * Ridge Regression: **L2 regularization** -- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
    * Lasso Regression: **L1 regularization** -- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    * ElasticNet Regression: **L1 and L2 Regularization** -- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html

* Linear Classification
    * Binary Classification: **Logistic Regression** -- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    * Multiclass Classification: **Softmax Regression** -- *use LogisticRegression() with multi_class="multinomial" and solver="lbfgs" hyperparameters.*

## Support Vector Machines (SVMs)
* SVM Classification
    * Linear: LinearSVC -- https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    * Non-Linear: SVC -- https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
* SVM Regression
    * Linear: LinearSVR -- https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html
    * Non-Linear: SVR -- https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
* Online SVMs
    * Linear -- SGD Classifier with ```loss='hinge' and alpha=(1/m*C)```
    * Non-Linear -- Neural Nets preferred over SVMs
