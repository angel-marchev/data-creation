# Some ideas on data synthesis & Data creation

# Angel Marchev

![](img/data_synth0.png)

<hr>

## What?

<span style="color:#000000">\(</span>  <span style="color:#374151">What is Synthetic Data?\)</span>

* mimics real-world data

* created, rather than collected

* can be generated in large quantities

* can be generated with specific properties

* may not always accurately reflect the complexities and variability of real-world data, so its results should be used with caution

<hr>

## WHY?

<span style="color:#000000">\(Why create data as means for data analysis?\)</span>

* imbalanced data (incl. short dataset) => oversampling, validation data

* feature engineering (incl. non-linear transform) => transformation

* missing data => partial imputation

* data privacy regulations => synthetic dataset

* lack of real data => synthetic dataset

* cost savings => synthetic dataset

(main emphasis today is synthesis of structured data, but some imputation is also within the scope due to methodical)

<hr>

<img src="img/data_synth28.png" width="600"/>

<hr>

## Data imputation

* various reasons for missing data, but main reasoning for imputation is if the missing data is related to other variables

![](img/data_synth30.jpg) <img src="img/data_synth29.jpg" width="200"/>

* Blue points are observations whereas red points are missing observations in the y-variable; statistics for complete data (blue and red combined) are slope (b) = 1, standard error (se) = 0.05 and R2 = 0.5. Assuming observations in the x-variable are complete, (a) represents missing at random (MAR), (b) represents missing not at random (MNAR) and (c) represents missing completely at random (MCAR). For the observed data (blue points), the estimated slope, se and R2, are (a) b = 0.86, se = 0.11, R2 = 0.29, (b) b = 0.432, se = 0.06, R2 = 0.23 and (c) b = 0.957, se = 0.07, R2 = 0.49.

<hr>

## Imputation methods

<table>
  <tr>
    <th> <ul>
     <li>inference based (crosslinked) modeling (horizontal && vertical)</li>
     <li>vs</li>
     <li>single variable imputation (vertical)</li>
     </ul>
</th>
    <th><img src="img/data_synth31.png" width="600"/></th>
  </tr>
</table>

<hr>

## Data imputation tricks

![](img/data_synth32.png)

<span style="color:#000000">distribution preservation noise</span>

<span style="color:#000000">“</span>  <span style="color:#000000">unknown” class</span>

<span style="color:#000000">mean by subclass instead of the whole variable</span>

<span style="color:#000000">The orthogonal view</span>

<hr>

## Distribution altering single variable imputation

![](img/data_synth57.png)

## Random Naive Oversampling

![](img/data_synth33.png)

<span style="color:#000000">randomly duplicating instances from the minority class until it is balanced with the majority class</span>

<span style="color:#000000">For example\, if the minority class has only 30% of the instances in a dataset\, then random naive oversampling would involve duplicating instances from the minority class 3 times</span>

<span style="color:#000000">it does not take into account the relationships between the features and the class labels \(hence called “naive”\)\.</span>

<span style="color:#000000">The simplest possible technique</span>

<span style="color:#000000">can lead to overfitting\, where the model trains to replicate too closely the repeated class\.</span>

![](img/data_synth34.png)

<hr>

## SMOTE and Variants

<span style="color:#000000">SMOTE \(Synthetic Minority Over\-sampling Technique\) is an algorithm for generating synthetic data\, specifically designed to address the problem of imbalanced datasets</span>

<span style="color:#000000">In a binary classification problem\, SMOTE generates synthetic samples of the minority class by interpolating between existing minority class samples\. </span>

<span style="color:#000000">Preserves the intrinsic characteristics of the minority class\. </span>

![](img/data_synth35.png)

![](img/data_synth36.png)

<hr>

<span style="color:#000000"> __SMOTE__ </span>  <span style="color:#000000"> \(and its many variations\) use the same algorithm to generate new samples\. Considering a sample </span>  <span style="color:#000000"> __x__ </span>  <span style="color:#000000"> _i_ </span>  <span style="color:#000000">\, a new sample </span>  <span style="color:#000000"> __x__ </span>  <span style="color:#000000"> _new_ </span>  <span style="color:#000000"> will be generated considering its k\- neareast\-neighbors \(corresponding to </span>  <span style="color:#000000"> __k__ </span>  <span style="color:#000000"> _neighbors_ </span>  <span style="color:#000000">\)\. </span>

<span style="color:#000000">For instance\, the 3 nearest\-neighbors are included in the blue circle as illustrated in the figure below\. Then\, one of these nearest\-neighbors </span>  <span style="color:#000000"> __x__ </span>  <span style="color:#000000"> _zi_ </span>  <span style="color:#000000"> is selected and a sample is generated as follows:</span>

<span style="color:#000000">   </span>  <span style="color:#000000"> __x__ </span>  <span style="color:#000000"> _new_ </span>  <span style="color:#000000"> __ = x__ </span>  <span style="color:#000000"> _i_ </span>  <span style="color:#000000"> __ \+ __ </span>  <span style="color:#000000"> __λ \.__ </span>  <span style="color:#000000"> __ \(x__ </span>  <span style="color:#000000"> _zi_ </span>  <span style="color:#000000"> __ \- x__ </span>  <span style="color:#000000"> _i_ </span>  <span style="color:#000000"> __\)__ </span>

<span style="color:#000000">where </span>  <span style="color:#000000"> __λ__ </span>  <span style="color:#000000"> is a random number in the range \[0\, 1\]\. This interpolation will create a sample on the line between </span>  <span style="color:#000000"> __x__ </span>  <span style="color:#000000"> _i_ </span>  <span style="color:#000000"> and </span>  <span style="color:#000000"> __x__ </span>  <span style="color:#000000"> _zi_ </span>  <span style="color:#000000"> as illustrated in the image\.</span>

<span style="color:#000000">SMOTE: This is the original implementation of SMOTE\, which generates synthetic samples by interpolating between pairs of minority class samples\. Specifically\, it selects a random minority class sample and its k\-nearest neighbors\, then generates new samples by interpolating between them\.</span>

<span style="color:#000000">Borderline\-SMOTE: This variant of SMOTE generates synthetic samples for the borderline instances of the minority class\. It identifies the minority class instances that are near the decision boundary of the classifier and applies SMOTE only to those samples\.</span>

<span style="color:#000000">Adaptive Synthetic Sampling \(ADASYN\): ADASYN is another variant of SMOTE that adapts the number of synthetic samples based on the density of the data distribution\. It generates more synthetic samples for minority class instances that are harder to learn\, thus reducing the bias in the classifier\.</span>

<span style="color:#000000">Safe\-Level SMOTE: This is a variant of SMOTE that considers the distribution of the majority class and generates synthetic samples based on the safe\-level\, which is the difference between the minority and majority class densities\.</span>

<span style="color:#000000">G\-SMOTE: This is a geometric variant of SMOTE that generates synthetic samples by extrapolating from the line segments joining k\-nearest neighbors\. It uses the geometry of the minority class samples to generate more realistic synthetic samples\.</span>

<span style="color:#000000">K\-Means SMOTE: This variant of SMOTE uses k\-means clustering to generate synthetic samples for the minority class\. It first clusters the minority class samples using k\-means\, then generates synthetic samples for each cluster\.</span>

<span style="color:#000000">The classifiers trained on synthetic examples generalize well\.</span>

<span style="color:#000000">The classifiers Identify the minority class well \(True Negatives\)\.</span>

<span style="color:#000000">They have fewer False Positives compared to undersampling\.</span>

<span style="color:#000000"> _Advantages _ </span>

<span style="color:#000000">It improves the overfitting caused by random oversampling as synthetic examples are generated rather than a copy of existing examples\.</span>

<span style="color:#000000">No loss of information\.</span>

<span style="color:#000000">It’s simple\.</span>

<span style="color:#000000"> _Disadvantages _ </span>

<span style="color:#000000">While generating synthetic examples\, SMOTE does not take into consideration neighboring examples that can be from other classes\. This can increase the overlapping of classes and can introduce additional noise\.</span>

<span style="color:#000000">SMOTE is not very practical for high\-dimensional data\.</span>

<hr>

## Random numbers generators

![](img/data_synth37.png)

<hr>

## Monte-Carlo simulation

![](img/data_synth38.png)

<span style="color:#374151">randomly generate many possible outcomes and use accumulation of these outcomes as an estimate of the expected value\.</span>

<span style="color:#000000">Relays on Law of large numbers </span>

<span style="color:#374151">Useful when it is difficult or impossible to solve a problem analytically\. The method can provide good results even when the underlying system is complex or poorly understood\. </span>

![](img/data_synth39.png)

<hr>

## Statistical distributions

![](img/data_synth44.png)

<span style="color:#000000">>100s</span>

<span style="color:#000000">Each with its own set of parameters</span>

<span style="color:#000000">Method of moments \(or GMM\)</span>

<span style="color:#000000">for distribution fitting</span>

<hr>

## Generating synthetic observations

<span style="color:#000000">\(Methodological background \)</span>

<span style="color:#000000">===Random===</span>

<span style="color:#000000">Monte\-Carlo simulation</span>

<span style="color:#000000">GAN</span>

<span style="color:#000000">===Non\-random===</span>

<span style="color:#000000">Inverse copula sampling</span>

<span style="color:#000000">Cholesky decomposition</span>

<span style="color:#374151">randomly sampling from a probability distribution</span>

<span style="color:#374151">repeat the sampling process multiple times to accumulate many simulated data points</span>

<span style="color:#374151">Analyze the generated synthetic data to validate that it resembles the distribution of the original data</span>

<span style="color:#000000">a type of deep learning architecture</span>

<span style="color:#000000">two neural networks: a generator and a discriminator</span>

<span style="color:#000000">trained together in a zero\-sum game</span>

<span style="color:#000000">copula is a function that describes a joint distribution of a set of variables</span>

<span style="color:#000000">modeling the dependence structure between variables in the original data and preserve it</span>

<span style="color:#000000">Cholesky decomposition is a factorization of a positive\-definite matrix</span>

<span style="color:#000000">modeling the covariance structure between variables in the original data</span>

<span style="color:#000000">\(Methodological framework\)</span>

<span style="color:#000000">\(opt\.\) Analyze for minimum needed features</span>

<span style="color:#000000">Data generation by \(chosen\) method</span>

<span style="color:#000000">Horizontal synchronization</span>

<span style="color:#000000">Feature \(vertical\) valida</span>

![](img/data_synth45.png)

<hr>

## Use case: only distributions

* <span style="color:#000000">IF Separate distributions for every variable:</span>
  * <span style="color:#000000">are available</span>
  * <span style="color:#000000">could be derived from small representative sample</span>
  * <span style="color:#000000">could be assumed</span>
* <span style="color:#000000">AND business rules have to be derived</span>
* <span style="color:#000000">THEN</span>
  * <span style="color:#000000">Feature\-wise Monte Carlo simulation</span>
  * <span style="color:#000000">Record\-wise business rules filtering</span>

![](img/data_synth46.png)

<hr>

## Use case: distributions & correlations

* <span style="color:#000000">IF Separate distributions for every variable:</span>
  * <span style="color:#000000">are available</span>
  * <span style="color:#000000">could be derived from small representative sample</span>
  * <span style="color:#000000">could be assumed</span>
* <span style="color:#000000">AND correlations between each variable pair </span>
* <span style="color:#000000">are available</span>
* <span style="color:#000000">THEN</span>
  * <span style="color:#000000">Correlation matrix</span>
  * <span style="color:#000000">Cholesky decomposition</span>
  * <span style="color:#000000">Monte Carlo simulation</span>
  * <span style="color:#000000">Pair wise product </span>

![](img/data_synth47.png)

![](img/data_synth48.png)

<hr>

## Use case: joint distribution

* <span style="color:#000000">IF Multi\-variate joint distribution \(copula function\) is available</span>
* <span style="color:#000000">THEN</span>
  * <span style="color:#000000">Use copula function to generate simulated normalized sample vectors</span>
  * <span style="color:#000000">De\-normalize using s</span>  <span style="color:#202122">caling variables of each feature</span>

![](img/data_synth49.png)

![](img/data_synth50.png)

<hr>

## Use case: some real features

* <span style="color:#000000">IF partial real data set is available</span>
* <span style="color:#000000">AND business rules have to be derived</span>
* <span style="color:#000000">THEN</span>
  * <span style="color:#000000">Derive new variables using:</span>
    * <span style="color:#000000">Business rules</span>
    * <span style="color:#000000">Scaling weights to the real variables</span>
  * <span style="color:#000000">Record\-wise business rules filtering</span>

![](img/data_synth51.png)

![](img/data_synth52.png)

<hr>

## Use case: parts of data sets

* <span style="color:#000000">IF parts of various data sets are available:</span>
  * <span style="color:#000000">From different sources</span>
  * <span style="color:#000000">Real data </span>
  * <span style="color:#000000">Synthetic data</span>
* <span style="color:#000000">AND there is overlap of at least two variables for every partial data set</span>
* <span style="color:#000000">THEN</span>
  * <span style="color:#000000">Use fuzzy matching algorithm to stich together the data sets on the overlapping \(key\) values</span>
    * <span style="color:#000000">They might need to be recoded</span>
  * <span style="color:#000000">In case of inconclusive fuzzy matching score or duplicates\, pick randomly \(probabilistic concatenation\)</span>

![](img/data_synth53.png)

<hr>

## Use case: full data sets

![](img/data_synth54.png)

* <span style="color:#000000">IF full data set is available:</span>
  * <span style="color:#000000">confidential</span>
  * <span style="color:#000000">sensitive</span>
  * <span style="color:#000000">without license</span>
* <span style="color:#000000">THEN</span>
  * <span style="color:#000000">Generate initial random data set</span>
  * <span style="color:#000000">Train Generative adversarial network on the real data set to generate synthetic data</span>
    * <span style="color:#000000">Using the generative ANN\, generate values of simulated data </span>
    * <span style="color:#000000">Using discriminative ANN\, discriminate real data from simulated data</span>
    * <span style="color:#000000">If discrimination is successful feed the result to the generative ANN for next simulation</span>
    * <span style="color:#000000">Else feed the result to discriminative ANN to improve discrimination next time</span>

<hr>

![](img/data_synth55.png)

Less data limitations & challenges => => => => =>

<hr>

## Sci-kit learn functions

Scikit-learn offers several functions for synthesizing datasets, each tailored for different types of data and machine learning problems. These functions are useful for generating data for experimentation, education, and benchmarking algorithms. Here are some of the main functions provided by Scikit-learn for data synthesis. Each of these functions allows you to specify parameters such as the number of samples, features, informative features, and levels of noise, giving you a lot of flexibility in generating datasets that fit your specific needs for testing and training machine learning models.
    

### 0. **`make_classification`**
This function is highly useful for generating a random n-class classification problem, which helps in testing classification algorithms or for illustrative purposes. 


```python
from sklearn.datasets import make_classification
X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9], n_informative=2, n_redundant=0, 
    flip_y=0, n_features=2, n_clusters_per_class=1, n_samples=100, random_state=42)
```

### 1. **`make_regression`**
This function generates a regression problem with a given number of features, noise level, and other characteristics. It's useful for testing regression algorithms.


```python
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=2, noise=0.1)
```

### 2. **`make_blobs`**
Useful for generating isotropic Gaussian blobs for clustering. It can simulate datasets with specific centers or cluster standard deviations.


```python
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers=3, n_features=2, cluster_std=1.5)
```

### 3. **`make_moons`**
This function generates a toy dataset of two interleaving half circles (a simple case of non-linearly separable dataset), which is useful for testing algorithms that can capture complex non-linear relationships.


```python
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, noise=0.1)
```

### 4. **`make_circles`**
Similar to `make_moons`, but this function generates a larger circle containing a smaller circle in 2d. It is another example of a dataset that is useful for evaluating the performance of clustering and classification algorithms.


```python
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=100, factor=0.5, noise=0.05)
```

### 5. **`make_multilabel_classification`**
This function creates a random multilabel classification problem, useful for testing multilabel classification algorithms. It generates samples that simulate the dependencies between classes for a more realistic data structure.


```python
from sklearn.datasets import make_multilabel_classification

X, y = make_multilabel_classification(n_samples=100, n_classes=3, n_labels=2)
```

### 6. **`make_hastie_10_2`**, **`make_friedman1`**, **`make_friedman2`**, **`make_friedman3`**
These functions generate data for regression problems with features that influence the target in non-linear and interaction ways, based on the work of statisticians Hastie and Friedman.


```python
from sklearn.datasets import make_friedman1

X, y = make_friedman1(n_samples=100, n_features=10, noise=0.0)

```

<hr>

## The Case

<span style="color:#0000FF"> _[https://github\.com/angel\-marchev/case\-cold\-start\-modeling](https://github.com/angel-marchev/case-cold-start-modeling)_ </span>

![](img/data_synth56.png)

<hr>

## See also

<span style="color:#000000">How to Use Synthetic and Simulated Data Effectively</span>

<span style="color:#0000FF"> _[https://towardsdatascience\.com/how\-to\-use\-synthetic\-and\-simulated\-data\-effectively\-04d8582b6f88](https://towardsdatascience.com/how-to-use-synthetic-and-simulated-data-effectively-04d8582b6f88)_ </span>

<span style="color:#000000">Best Practices and Lessons Learned on Synthetic Data for Language Models</span>

<span style="color:#0000FF"> _[https://arxiv\.org/pdf/2404\.07503\.pdf](https://arxiv.org/pdf/2404.07503.pdf)_ </span>

<span style="color:#000000">Google AI Introduces CodecLM: A Machine Learning Framework for Generating High\-Quality Synthetic Data for LLM Alignment</span>

<span style="color:#0000FF"> _[https://www\.marktechpost\.com/2024/04/13/google\-ai\-introduces\-codeclm\-a\-machine\-learning\-framework\-for\-generating\-high\-quality\-synthetic\-data\-for\-llm\-alignment/](https://www.marktechpost.com/2024/04/13/google-ai-introduces-codeclm-a-machine-learning-framework-for-generating-high-quality-synthetic-data-for-llm-alignment/)_ </span>

<hr>

## List with publications:

1\. Marchev\, A\.\, Marchev\, V\.\, 2024\, Automated Algorithm for Multi\-variate Data Synthesis with Cholesky Decomposition\, ICACS 2023: the 7th International Conference on Algorithms\, Computing and Systems\, Larissa Greece\, Association for Computing Machinery\, New York\, pp\. 1 – 6\, ISBN:979\-8\-4007\-0909\-8\, DOI: 10\.1145/3631908\.3631909;

2\. Marchev\, V\.\, Marchev\, A\.\, Piryankova\, M\.\, Masarliev\, D\.\, Mitkov\, V\.\, 2023\, Synthesizing an anonymized multidimensional dataset featuring financial\, economic\, demographic\, and personal traits data\, VSIM\, vol\. 19\, no\. 1\, 2023\, ISSN 1314\-0582;

3\. Marchev\, A\.\, Marchev\, V\.\, 2022\. Synthesizing multi\-dimensional personal data sets\. AIP Conference Proceedings; 2505 \(1\): 020012\, DOI: 10\.1063/5\.0100615;

4\. Marchev\, V\, Marchev\, A\.\, 2021\, “Methods for Simulating Multi\-dimensional Data for Financial Services Recommendation”\, Bulgarian Economic Papers\, Center for economic theories and policies\, ISSN: 2367\-7082\, BEP 02\-2021\, Feb\. 2021\, BEP;

5\. Марчев В\.\, Марчев\, А\.\, 2020\, Симулация на многокритериална база от данни за банкови услуги\. Алгоритъм и бизнес логика\, „Новите информационни технологии и големите данни: възможности и перспективи при анализите и управленските решения в бизнеса\, икономиката и социалната сфера;

6\. Marchev\, V\. Marchev\, A\.\, 2024\, Anonymizing Personal Information Using Distribution\-based Data Synthesis\, in publishing\.

7\. Lyubchev\, D\.\, Marchev\, A\.\, Marchev\, V\., Inverse copula sampling for multi-dimensional data synthesis\(2024\)

8\. Marchev\, V\. Marchev\, A\., Methodological Considerations for Anonymizing Tabular Data Using Generative Adversarial Networks\(2025\)
