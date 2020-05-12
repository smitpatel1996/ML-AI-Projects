# ML Model Cheat-Sheet
### ***Scikit-Learn UserGuide*** -- https://scikit-learn.org/stable/user_guide.html

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
    * To use soft voting: set hyperparameter ```voting='soft'``` if all the predictors have a *predict_proba() method.*
* Bagging and Pasting
    * **BaggingClassifier** -- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
    * **BaggingRegressor** -- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html
    * To use pasting ensemble: set hyperparameter ```bootstrap=False```.
* Boosting
    * AdaBoost
        * **AdaBoostClassifier** -- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
        * **AdaBoostRegressor** -- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
    * Gradient Boosting
        * **GradientBoostingClassifier** -- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
        * **GradientBoostingRegressor** -- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
* Stacking
    * *Sklearn Implementation Unavailabel* -- Child Predictors predict values which are used by the Parent Predictor (called a blender, or a meta learner) as inputs to make final prediction. (Basic Concept) -- *Can be used to stack bagging, boosting and other ensemble methods for even better performances possibly.*

## Dimensionality Reduction
* Projection Technique
    * **PCA** -- Default, Linear Datasets -- https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    * **Incremental PCA** -- Large Dataset, Online -- https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html
    * **Randomized PCA** -- new D << old D, Faster -- in PCA use ```svd_solver="randomized"```
    * **Kernal PCA** -- Nonlinear Datasets -- https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html
* Manifold Learning
    * **LLE** -- NLDR -- https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html
    * **MDS** -- NLDR -- https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html
    * **Isomap** -- NLDR -- https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html
    * **t-SNE** -- NLDR, Visualizing Clusters -- https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    * **LDA** -- DR before Classification -- https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html

# Widely Used Scikit-Learn Modules
### ***Scikit-Learn API Reference*** -- https://scikit-learn.org/stable/modules/classes.html

* **sklearn.base** -- Base classes for all estimators.
* **sklearn.cluster** -- contains popular Unsupervised Clustering Algorithms.
* **sklearn.compose** -- used for building composite models with transformers.
* **sklearn.decomposition** -- includes dimensionality reduction techniques.
* **sklearn.discriminant_analysis** -- includes Linear Discriminant Analysis and Quadratic Discriminant Analysis.
* **sklearn.ensemble** -- includes ensemble-based methods for classification, regression and anomaly detection.
* **sklearn.feature_extraction** -- deals with feature extraction from raw data (images and text).
* **sklearn.feature_selection** -- implements feature selection algorithms.
* **sklearn.impute** -- missing value imputation.
* **sklearn.linear_model** -- implements linear models.
* **sklearn.manifold** -- implements manifold learning algorithms.
* **sklearn.metrics** -- includes score functions, performance metrics and pairwise metrics and distance computations.
* **sklearn.model_selection** -- tools to help select best model.
* **sklearn.multiclass** -- implements multiclass and multilabel learning algorithms.
* **sklearn.multioutput** -- implements multioutput regression and classification.
* **sklearn.naive_bayes** -- implements Naive Bayes algorithms.
* **sklearn.neighbors** -- implements the k-nearest neighbors algorithm.
* **sklearn.neural_network** -- includes models based on neural networks.
* **sklearn.pipeline** -- implements utilities to build a composite estimator, as a chain of transforms and estimators.
* **sklearn.preprocessing** -- includes scaling, centering, normalization, binarization methods.
* **sklearn.svm** -- includes Support Vector Machine algorithms.
* **sklearn.tree** -- includes decision tree-based models for classification and regression.
