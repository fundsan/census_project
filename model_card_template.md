# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
– Developed by Jake Fund
– September 2, 2021
– v1
– Logistic Regression
– Model needs one-hot encoder and labelizer transformers
– Citation details:
* https://arxiv.org/pdf/1810.03993.pdf
* @article{2018aequitas,
     title={Aequitas: A Bias and Fairness Audit Toolkit},
     author={Saleiro, Pedro and Kuester, Benedict and Stevens, Abby and Anisfeld, Ari and Hinkson, Loren and London, Jesse and Ghani, Rayid}, journal={arXiv preprint arXiv:1811.05577}, year={2018}}

## Intended Use
– Primary intended uses:Predicting salary based on demographic data

## Training Data
– Datasets: 80% of https://archive.ics.uci.edu/ml/datasets/census+income 
– Preprocessing: categorical data is put into one hot encoding
## Evaluation Data
– Datasets:20% of https://archive.ics.uci.edu/ml/datasets/census+income 
– Motivation:
– Preprocessing:

## Metrics
– Model performance measures
– Decision thresholds
– Variation approache

## Ethical Considerations
– Unitary results
– Intersectional results:
![alt text](images/slice_performance_output.png)

## Caveats and Recommendations
