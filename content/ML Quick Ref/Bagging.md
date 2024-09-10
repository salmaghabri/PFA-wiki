- ensemble learning method that combines multiple models to improve the stability and accuracy of machine learning algorithms.
## Key Components

1. **Bootstrap Sampling**: Bagging involves creating multiple samples of the training data by randomly selecting rows with replacement. This ensures that each model is trained on a different subset of the data.
2. **Base Model**: The base model used in bagging is typically an unpruned decision tree. Each model is trained on a different bootstrap sample of the data.
3. **Aggregation**: The predictions made by each model are then combined using simple statistics, such as voting or averaging.
## Steps

1. **Create Bootstrap Samples**: Create multiple bootstrap samples of the training data by randomly selecting rows with replacement.
2.  **Train Base Models**: Train multiple base models (usually decision trees) on each bootstrap sample.
3.  **Combine Predictions**: Combine the predictions made by each base model using simple statistics, such as voting or averaging.