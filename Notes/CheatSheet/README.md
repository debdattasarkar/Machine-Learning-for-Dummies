![ViewCount](https://views.whatilearened.today/views/github/debdattasarkar/Machine-Learning-for-Dummies.svg?cache=remove)
![GitHub top language](https://img.shields.io/github/languages/top/debdattasarkar/Machine-Learning-for-Dummies?style=flat)
![GitHub language count](https://img.shields.io/github/languages/count/debdattasarkar/Machine-Learning-for-Dummies?style=flat)
![Stars Badge](https://img.shields.io/github/stars/debdattasarkar/Machine-Learning-for-Dummies?style=flat)
![Forks Badge](https://img.shields.io/github/forks/debdattasarkar/Machine-Learning-for-Dummies?style=flat)

# Machine Learning For Dummies Cheat Sheet
### By John Paul Mueller, Luca Massaron

Cheat Sheet downloaded from "www.dummies.com".

<div>
<p>Machine learning is an incredible technology that you use more often than you think today and that has the potential to do even more tomorrow. The interesting thing about machine learning is that <a href="https://www.dummies.com/programming/python/python-all-in-one-for-dummies-cheat-sheet/">Python</a> makes the task easier than most people realize because it comes with a lot of built-in and extended support (through the use of libraries, datasets, and other resources). With that in mind, this Cheat Sheet helps you access the most commonly needed reminders for making your machine learning experience fast and easy.</p>

<div class="cheat-sheet-section">
<h2 id="tab1">Locate the Algorithm You Need</h2>
<p>Machine learning requires the use of a large number of algorithms to perform various tasks. However, finding the specific algorithm you want to know about could be difficult. The following table provides you with an online location for information about the most common algorithms.</p>
<table width="100%">
<tbody>
<tr>
<td width="10%"><strong>Algorithm</strong></td>
<td width="10%"><strong>Type</strong></td>
<td width="60%"><strong>Python/R URL</strong></td>
</tr>
<tr>
<td width="10%">Naïve Bayes</td>
<td width="10%">Supervised classification, online learning</td>
<td width="60%"><a href="https://scikit-learn.org/stable/modules/naive_bayes.html">https://scikit-learn.org/stable/modules/naive_bayes.html</a></td>
</tr>
<tr>
<td width="10%">PCA</td>
<td width="10%">Unsupervised</td>
<td width="60%"><a href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html">https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html</a></td>
</tr>
<tr>
<td width="10%">SVD</td>
<td width="10%">Unsupervised</td>
<td width="60%"><a href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html">https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html</a></td>
</tr>
<tr>
<td width="10%">K-means</td>
<td width="10%">Unsupervised</td>
<td width="60%"><a href="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html">https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html</a></td>
</tr>
<tr>
<td width="10%">K-Nearest Neighbors</td>
<td width="10%">Supervised regression and classification</td>
<td width="60%"><a href="https://scikit-learn.org/stable/modules/neighbors.html">https://scikit-learn.org/stable/modules/neighbors.html</a></td>
</tr>
<tr>
<td width="10%">Linear Regression</td>
<td width="10%">Supervised regression, online learning</td>
<td width="60%"><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html</a></td>
</tr>
<tr>
<td width="10%">Logistic Regression</td>
<td width="10%">Supervised classification, online learning</td>
<td width="60%"><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html">https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html</a></td>
</tr>
<tr>
<td width="10%">Neural Networks</td>
<td width="10%">Unsupervised Supervised regression and classification</td>
<td width="60%"><a href="https://scikit-learn.org/dev/modules/neural_networks_supervised.html">https://scikit-learn.org/dev/modules/neural_networks_supervised.html</a></td>
</tr>
<tr>
<td width="10%">Support Vector Machines</td>
<td width="10%">Supervised regression and classification</td>
<td width="60%"><a href="https://scikit-learn.org/stable/modules/svm.html">https://scikit-learn.org/stable/modules/svm.html</a></td>
</tr>
<tr>
<td width="10%">Adaboost</td>
<td width="10%">Supervised classification</td>
<td width="60%"><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html">https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html</a></td>
</tr>
<tr>
<td width="10%">Gradient Boosting</td>
<td width="10%">Supervised regression and classification</td>
<td width="60%"><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html">https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html</a></td>
</tr>
<tr>
<td width="10%">Random Forest</td>
<td width="10%">Supervised regression and classification</td>
<td width="60%"><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html">https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html</a></td>
</tr>
</tbody>
</table>
</div>

<div class="cheat-sheet-section">
<h2 id="tab2">Choose the Right Algorithm</h2>
<p><em>Machine Learning For Dummies, 2nd Edition</em> discusses a lot of different algorithms, and it may seem at times as if it will never run out. The following table provides you with a quick summary of the strengths and weaknesses of the various algorithms.</p>
<table width="100%">
<tbody>
<tr>
<td><strong>Algorithm</strong></td>
<td width="25%"><strong>Best at</strong></td>
<td width="25%"><strong>Pros</strong></td>
<td width="25%"><strong>Cons</strong></td>
</tr>
<tr>
<td>Random Forest</td>
<td width="25%">
<ul>
<li>Apt at almost any machine learning problem</li>
<li>Bioinformatics</li>
</ul>
</td>
<td width="25%">
<ul>
<li>Can work in parallel</li>
<li>Seldom overfits</li>
<li>Automatically handles missing values if you impute using a special number</li>
<li>No need to transform any variable</li>
<li>No need to tweak parameters</li>
</ul>
</td>
<td width="25%">
<ul>
<li>Difficult to interpret</li>
<li>Weaker on regression when estimating values at the extremities of the distribution of response values</li>
<li>Biased in multiclass problems toward more frequent classes</li>
</ul>
</td>
</tr>
<tr>
<td>Gradient Boosting</td>
<td width="25%">
<ul>
<li>Apt at almost any machine learning problem</li>
<li>Search engines (solving the problem of learning to rank)</li>
</ul>
</td>
<td width="25%">
<ul>
<li>It can approximate most nonlinear function</li>
<li>Best in class predictor</li>
<li>Automatically handles missing values</li>
<li>No need to transform any variable</li>
</ul>
</td>
<td width="25%">
<ul>
<li>It can overfit if run for too many iterations</li>
<li>Sensitive to noisy data and outliers</li>
<li>Doesn’t work at its best without parameter tuning</li>
</ul>
</td>
</tr>
<tr>
<td>Linear regression</td>
<td width="25%">* Baseline predictions<br>
* Econometric predictions<br>
* Modelling marketing responses</td>
<td width="25%">* Simple to understand and explain<br>
* It seldom overfits<br>
* Using L1 &amp; L2 regularization is effective in feature selection<br>
* Fast to train<br>
* Easy to train on big data thanks to its stochastic version</td>
<td width="25%">* You have to work hard to make it fit nonlinear functions<br>
* Can suffer from outliers</td>
</tr>
<tr>
<td>Support Vector Machines</td>
<td width="25%">
<ul>
<li>Character recognition</li>
<li>Image recognition</li>
<li>Text classification</li>
</ul>
</td>
<td width="25%">
<ul>
<li>Automatic non-linear feature creation</li>
<li>Can approximate complex non-linear functions</li>
<li>Works only with a portion of the examples (the support vectors)</li>
</ul>
</td>
<td width="25%">
<ul>
<li>Difficult to interpret when applying  non-linear kernels</li>
<li>Suffers from too many examples, after 10,000 examples it starts taking too long to train</li>
</ul>
</td>
</tr>
<tr>
<td>K-Nearest Neighbors</td>
<td width="25%">
<ul>
<li>Computer vision</li>
<li>Multilabel tagging</li>
<li>Recommender systems</li>
<li>Spell checking problems</li>
</ul>
</td>
<td width="25%">
<ul>
<li>Fast, lazy training</li>
<li>Can naturally handle extreme multiclass problems (like tagging text)</li>
</ul>
</td>
<td width="25%">
<ul>
<li>Slow and cumbersome in the predicting phase</li>
<li>Can fail to predict correctly due to the curse of dimensionality</li>
</ul>
</td>
</tr>
<tr>
<td>Adaboost</td>
<td width="25%">
<ul>
<li>Face detection</li>
</ul>
</td>
<td width="25%">
<ul>
<li>Automatically handles missing values</li>
<li>No need to transform any variable</li>
<li>It doesn’t overfit easily</li>
<li>Few parameters to tweak</li>
<li>It can leverage many different weak-learners</li>
</ul>
</td>
<td width="25%">
<ul>
<li>Sensitive to noisy data and outliers</li>
<li>Never the best in class predictions</li>
</ul>
</td>
</tr>
<tr>
<td>Naive Bayes</td>
<td width="25%">
<ul>
<li>Face recognition</li>
<li>Sentiment analysis</li>
<li>Spam detection</li>
<li>Text classification</li>
</ul>
</td>
<td width="25%">
<ul>
<li>Easy and fast to implement, doesn’t require too much memory and can be used for online learning</li>
<li>Easy to understand</li>
<li>Takes into account prior knowledge</li>
</ul>
</td>
<td width="25%">
<ul>
<li>Strong and unrealistic feature independence assumptions</li>
<li>Fails estimating rare occurrences</li>
<li>Suffers from irrelevant features</li>
</ul>
</td>
</tr>
<tr>
<td>Neural Networks</td>
<td width="25%">
<ul>
<li>Image recognition</li>
<li>Language recognition and translation</li>
<li>Speech recognition</li>
<li>Vision recognition</li>
</ul>
</td>
<td width="25%">
<ul>
<li>It can approximate any non-linear function</li>
<li>Robust to outliers</li>
<li>It can work with image, text and sound data</li>
</ul>
</td>
<td width="25%">
<ul>
<li>It requires you to define a network architecture</li>
<li>Difficult to tune because of too many parameters and you have also to decide the architecture of the network</li>
<li>Difficult to interpret</li>
<li>Easy to overfit</li>
</ul>
</td>
</tr>
<tr>
<td>Logistic regression</td>
<td width="25%">
<ul>
<li>Ordering results by probability</li>
<li>Modelling marketing responses</li>
</ul>
</td>
<td width="25%">
<ul>
<li>Simple to understand and explain</li>
<li>It seldom overfits</li>
<li>Using L1 &amp; L2 regularization is effective in feature selection</li>
<li>The best algorithm for predicting probabilities of an event</li>
<li>Fast to train</li>
<li>Easy to train on big data thanks to its stochastic version</li>
</ul>
</td>
<td width="25%">
<ul>
<li>You have to work hard to make it fit non-linear functions</li>
<li>Can suffer from outliers</li>
</ul>
</td>
</tr>
<tr>
<td>SVD</td>
<td width="25%">
<ul>
<li>Recommender systems</li>
</ul>
</td>
<td width="25%">
<ul>
<li>Can restructure data in a meaningful way</li>
</ul>
</td>
<td width="25%">
<ul>
<li>Difficult to understand why data has been restructured in a certain way</li>
</ul>
</td>
</tr>
<tr>
<td>PCA</td>
<td width="25%">
<ul>
<li>Removing collinearity</li>
<li>Reducing dimensions of the dataset</li>
</ul>
</td>
<td width="25%">
<ul>
<li>Can reduce data dimensionality</li>
</ul>
</td>
<td width="25%">
<ul>
<li>Implies strong linear assumptions (components are a weighted summations of features)</li>
</ul>
</td>
</tr>
<tr>
<td>K-means</td>
<td width="25%">
<ul>
<li>Segmentation</li>
</ul>
</td>
<td width="25%">
<ul>
<li>Fast in finding clusters</li>
<li>Can detect outliers in multiple dimensions</li>
</ul>
</td>
<td width="25%">
<ul>
<li>Suffers from multicollinearity</li>
<li>Clusters are spherical, can’t detect groups of other shape</li>
<li>Unstable solutions, depends on initialization</li>
</ul>
</td>
</tr>
</tbody>
</table>
</div>

<div class="cheat-sheet-section">
<h2 id="tab3">Get the Right Package</h2>
<p>When working with Python, you gain the benefit of not having to reinvent the wheel when it comes to algorithms. There is a package available to meet your specific needs—you just need to know which one to use. The following table provides you with a listing of common Python packages. When you want to perform any algorithm-related task, simply load the package needed for that task into your programming environment.</p>
<ul>
<li><strong>Adaboost: </strong>ensemble.AdaBoostClassifier and sklearn.ensemble.AdaBoostRegressor</li>
<li><strong>Gradient Boosting: </strong>ensemble.GradientBoostingClassifier and sklearn.ensemble.GradientBoostingRegressor</li>
<li><strong>K-means: </strong>cluster.KMeans and sklearn.cluster.MiniBatchKMeans</li>
<li><strong>K-Nearest Neighbors: </strong>neighbors.KNeighborsClassifier and sklearn.neighbors.KNeighborsRegressor</li>
<li><strong>Linear regression: </strong>linear_model.LinearRegression, sklearn.linear_model.Ridge, sklearn.linear_model.Lasso, sklearn.linear_model.ElasticNet, and sklearn.linear_model.SGDRegressor</li>
<li><strong>Logistic regression: </strong>linear_model.LogisticRegression and sklearn.linear_model.SGDClassifier</li>
<li><strong>Naive Bayes: </strong>naive_bayes.GaussianNB. sklearn.naive_bayes.MultinomialNB, and sklearn.naive_bayes.BernoulliNB</li>
<li><strong>Neural Networks: </strong>keras</li>
<li>Principal Component Analysis (PCA): sklearn.decomposition.PCA</li>
<li><strong>Random Forest: </strong>ensemble.RandomForestClassifier. sklearn.ensemble.RandomForestRegressor, sklearn.ensemble.ExtraTreesClassifier, and sklearn.ensemble.ExtraTreesRegressor</li>
<li><strong>Support Vector Machines (SVMs): </strong>svm.SVC, sklearn.svm.LinearSVC, sklearn.svm.NuSVC, sklearn.svm.SVR, sklearn.svm.LinearSVR, sklearn.svm.NuSVR, and sklearn.svm.OneClassSVM</li>
<li><strong>Singular Value Decomposition (SVD): </strong>decomposition.TruncatedSVD and sklearn.decomposition.NMF</li>
</ul></div>

<div class="cheat-sheet-section">
<h2 id="tab4">Differentiating Learning Types</h2>
<p>Algorithms are said to learn, but it’s important to know how they learn because they most definitely don’t learn in the same way that humans do. Learning comes in many different flavors, depending on the algorithm and its objectives. You can divide machine learning algorithms into three main groups based on their purpose:</p>
<ul>
<li><strong>Supervised learning:</strong> Occurs when an algorithm learns from example data and associated target responses that can consist of numeric values or string labels — such as classes or tags — in order to later predict the correct response when posed with new examples. The supervised approach is, indeed, similar to human learning under the supervision of a teacher. The teacher provides good examples for the student to memorize, and the student then derives general rules from these specific examples.</li>
<li><strong>Unsupervised learning:</strong> Occurs when an algorithm learns from plain examples without any associated response, leaving the algorithm to determine the data patterns on its own. This type of algorithm tends to restructure the data into something else, such as new data features that may represent a class or some new values helpful for additional analysis or for the training a predictive model.</li>
<li><strong>Reinforcement learning:</strong> Occurs when you sequentially present the algorithm with examples that lack labels, as in unsupervised learning. However, you accompany each example with positive or negative feedback according to the solution the algorithm proposes. Reinforcement learning is connected to applications for which the algorithm must make decisions (so that the product is prescriptive, not just descriptive, as in unsupervised learning), and the decisions bear consequences.</li>
</ul></div>

</div>


