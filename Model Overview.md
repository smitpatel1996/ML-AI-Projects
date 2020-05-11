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
    * Multiclass Classification: **Softmax Regression** -- *use LogisticRegression() with ```multi_class="multinomial" and solver="lbfgs"``` hyperparameters.*

## Support Vector Machines (SVMs)
* SVM Classification
    * Linear: **LinearSVC** -- https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    * Non-Linear: **SVC** -- https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
* SVM Regression
    * Linear: **LinearSVR** -- https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html
    * Non-Linear: **SVR** -- https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
* Online SVMs
    * Linear -- *use SGDClassifier() with ```loss='hinge' and alpha=(1/m*C)``` hyperparameters.*
    * Non-Linear -- Neural Nets preferred over SVMs

## Decision Trees
* Classification
    * **DecisionTreeClassifier** -- https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
* Regression
    * **DecisionTreeRegressor** -- https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
* Visualize Decision Tree
    * **Export Graphviz** -- https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html

## Random Forests
* Classification
    * **RandomForestClassifier** -- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    * **ExtraTreesClassifier** -- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
* Regression
    * **RandomForestRegressor** -- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    * **ExtraTreesRegressor** -- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html

## Ensemble Learning
* Voting Ensemble (Hard and Soft Voting)
    * **VotingClassifier** -- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
    * **VotingRegressor** -- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html
* Bagging and Pasting
    * **BaggingClassifier** -- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
    * **BaggingRegressor** -- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html
* Boosting
    * AdaBoost
        * **AdaBoostClassifier** -- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
        * **AdaBoostRegressor** -- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
    * Gradient Boosting
        * **GradientBoostingClassifier** -- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
        * **GradientBoostingRegressor** -- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
* Stacking
    * *Sklearn Implementation Unavailabel* -- Child Predictors predict values which are used by the Parent Predictor (called a blender, or a meta learner) as inputs to make final prediction. (Basic Concept) -- *Can be used to stack bagging, boosting and other ensemble methods for even better performances possibly.*