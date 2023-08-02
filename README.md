# Decision-Trees

Decision trees are a popular and interpretable machine learning algorithm used for both classification and regression tasks. They partition the feature space into subsets based on the values of input features, creating a tree-like structure of decisions and outcomes. Each internal node represents a decision based on a specific feature, while the leaf nodes represent predicted outcomes. Decision trees are easy to understand and visualize, but they can suffer from overfitting. Techniques like pruning and ensemble methods (Random Forests, Gradient Boosting) are often used to mitigate this issue.

## Key-Points

**1. Splitting Criteria:** Decision trees use various metrics (e.g., Gini impurity, entropy, mean squared error) to determine the best way to split the data at each node. The goal is to maximize homogeneity within each resulting subset.

**2. Greedy Approach:** Decision trees employ a greedy approach, making locally optimal decisions at each node. This can lead to suboptimal results at the global level, which is why pruning is often used to simplify the tree.

**3. Categorical and Numeric Features:** Decision trees can handle both categorical and numeric features. They perform binary splits for numeric features and multi-way splits for categorical features.

**4.Interpretability:** Decision trees are highly interpretable, as the path from the root to a leaf node represents a clear decision-making process based on feature values.

**5. Overfitting and Pruning:** Decision trees tend to overfit when they capture noise in the data. Pruning involves removing branches that do not significantly improve predictive accuracy, leading to a simpler and more generalizable tree.

**6. Ensemble Methods:** Random Forests and Gradient Boosting are ensemble methods that leverage multiple decision trees to improve predictive performance and reduce overfitting.

**7. Bias-Variance Trade-off:** Decision trees have a high variance and can be sensitive to small fluctuations in the data. Ensembling techniques help mitigate this by combining multiple trees.

**8. Missing Values:** Decision trees can handle missing values during the tree-building process by considering available features.

**9. Feature Importance:** Decision trees can provide insight into feature importance based on how often they are used for splitting and their contribution to reducing impurity.

## Splitting Criteria

Splitting conditions refer to the criteria used to determine how the data should be partitioned at each internal node of the tree. The goal of these conditions is to maximize the homogeneity (purity) of the resulting subsets, which ultimately leads to better classification or regression performance. Different splitting criteria are used for categorical and numerical features.

1. For Categorical Features: One common splitting condition for categorical features is based on the concept of Gini impurity or entropy.

**Gini Impurity:** Gini impurity measures the probability of misclassifying a randomly chosen element from a node. It ranges from 0 (perfectly pure, all elements belong to the same class) to 0.5 (completely impure, elements are evenly distributed among classes).
  * Formula: Gini(D) = 1 - Σ (p_i)^2, where p_i is the proportion of instances of class i in node D.
**Entropy:** Entropy measures the level of disorder or uncertainty in a node's class distribution.

  * Formula: Entropy(D) = - Σ (p_i * log2(p_i)), where p_i is the proportion of instances of class i in node D.
The splitting condition for categorical features involves evaluating the impurity or entropy of each potential split and choosing the one that results in the greatest reduction in impurity or entropy.

**Information Gain:** Information gain measures the reduction in entropy achieved by partitioning the data based on a specific feature. It quantifies how much uncertainty about the class labels is reduced after the split.
  * Formula: Information Gain(D, A) = Entropy(D) - Σ ((|D_v| / |D|) * Entropy(D_v)), where D is the current node, A is the feature being considered, D_v is the subset of instances associated with feature value v.

2. For Numerical Features: When dealing with numerical features, decision trees use binary splits based on threshold values.

**Mean Squared Error (MSE):** For regression tasks, the mean squared error is often used as the splitting criterion. It measures the average squared difference between the actual and predicted values.
  * Formula: MSE(D) = (1 / n) * Σ (y_i - ŷ)^2, where y_i is the actual value, ŷ is the predicted value, and n is the number of instances in node D.
* The splitting condition for numerical features involves evaluating the MSE reduction for all possible threshold values and selecting the one that maximizes the reduction.

**Information Gain on Threshold (Numeric Feature):** Similar to information gain for categorical features, information gain can be used for numerical features by considering threshold-based splits. It calculates the reduction in entropy achieved by partitioning the data into two subsets based on a threshold value.

In summary, splitting conditions in decision trees involve calculating impurity (Gini impurity or entropy) or error (MSE) measures for potential splits based on either categorical or numerical features. The split that results in the greatest reduction in impurity or error is chosen to partition the data at each internal node, leading to the construction of an effective decision tree.
