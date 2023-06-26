# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model is a Random Forest classifier with default hyperparameters.

## Intended Use
The model predicts whether income of a person exceeds $50K/yr given his/her status.

## Training Data
The training data cointains census data from the 1994 Census dataset. The training set contains 80% of the original data.

## Evaluation Data
Evaluation set contains 20% of the original data.

## Metrics
The model was evaluated on the following metrics: precision (0.73), recall (0.65) and Fbeta (0.69).

## Ethical Considerations
Given that the data contains census information on sex, race, etc. we need make sure the model is not bias for a certain group.

## Caveats and Recommendations
To further improve the performance, hyperparameter tunning (using K-fold cross validation) shold be considered. Furthermore, feature engineering can help boosting the performance.
