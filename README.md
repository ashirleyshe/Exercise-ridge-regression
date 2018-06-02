# Exercise-ridge-regression
### problem description
* Implement the “gradient_descent” function for ridge regression (i.e., linear regression with L2-norm as the regularization term)  
* Use your script to predict the target (Y1) of the Energy Efficiency dataset  
https://archive.ics.uci.edu/ml/datasets/Energy+efficiency
* Separate the data into training (50%) and test (50%) datasets.

### gradient descent for linear regression
Assumption: 𝑦 = 𝜃_0+𝜃_1*𝑥1  
1. Start with random values  
2. Slightly move 𝜃_0 and 𝜃_1 to reduce J(𝜃)  
3. Keep doing step 2 until converges
