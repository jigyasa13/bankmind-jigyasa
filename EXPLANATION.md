# Model Evaluation & Project Insights


### 1. What percentage of customers in your dataset have `y = yes`? What does this imbalance mean for how you'd evaluate a model?

As i had found the percentage of class didtribution in EDA, we can clearly see that 11.69% have 'y=yes' which means this is a imabalanced dataset.
Now, this tells us, that accuracy does not matter for this model, because the model can easily fake it. Like even if it classifies all the data as 'y=yes', it will still get 88.69% accuracy, but that is completely fake because it did not learn anything.

### 2. Which job category had the highest subscription rate? Does this make sense to you intuitively?

Based on this standard telemarketing bank dataset, **students and retired individuals** actually have the highest rate of conversion, even though management and blue-collar jobs are called more frequently.
Yes, it makes intuitive sense because both students and retired people are looking for safe, fixed financial returns (like a term deposit) to secure their savings, and they typically have more flexibility to pick up telemarketing calls.

### 3. Which feature had the highest importance in your tree-based model? Why do you think that is?

Features like **`poutcome_success`** (customers who said yes and actually bought the product in previous campaign) or **`month`** have the highest importance. 
This is because, if the customer had siad yes in previous campaign and they did buy the product, then they are highly likely to say yes agin, as now we have gained their trust.

### 4. Why is F1 a better metric than accuracy for this particular dataset?
As we can see this is a highly imbalanced datset, so even if the model does not learn anything and just predicts 'no' in all cases, still it will have 88.69% accuracy. This gives us a fake illuson that the model has performed well.
While the 'F1-score' balances both precision and recall, through which we can see how well it handles the minority "yes" cases. So, we have to focus more on F1 score.

### 5. Pick one of your 5 sample predictions. Do you actually agree with the model's call, given that customer's features? Walk through your thinking.
I took the last 2 cases where the model predicted yes, and among them is a 36 year old technician , who is single, and has a huge bank balance and has no loans. And i definitely agree with models call, because
a) he has huge bank balance 
b)no family to take care of and 
c)no loans at the moment 
So there's a high chance that he will say yes.
