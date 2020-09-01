Analysis of Car Accident
======
# 1.Introduction

## 1.1 Background 
  Car accident is one of the most important reason that cause people's dead every year.There are many reasons leading to the accident, 
  including external factors and internal factors. External factors such as bad weather, pedestrians who do not obey traffic rules, and 
  animals that appear suddenly. And the internal factors are mostly caused by improper driving, such as fatigue driving, drunk driving, 
  driving after taking drugs and so on. Therefore, the analysis of the causes of accidents can effectively prevent the occurrence of 
  similar accidents next time, so as to reduce casualties.
 
## 1.2 Business Problem
  Traffic accident data can help to find out the causes of most traffic accidents, including weather, speed, road conditions and so on. The aim of the project is to use the data and build a model. When there gives some data, like weather, light condition, road condition, the model will classify the data, and make prediction about its possible damage: property damage or injured people.
 # 2.Data section

 ## 2.1 Data sourse
 The dataset include 194673 records of car accident, and 37 attributes. The attributes are including weather condition, light condition, speed and the amount of injuried people,etc. From each row of data, we could know how the accident happen and how many people get injuired in the accident. By using the this kind of data, we cound do analysis and found out which factor is the most significant one and prevent it next time. However,some of the data is empty. ROADCOND: The condition of The road during The collision. This content has about 5,000 null values in the database. Considering that this factor is not of great help to the analysis of traffic accidents, this information was deleted from the database during the analysis process in order to better complete the project.

# 3.Methodology
First,is to clean and prepare the data in the data set. Then normalize the data and split it into train data and test data to find the accuracy of each model.
There are three kinds of methods: KNN, Decision Tree and Logistic Regression. Finally, use F1-score and Jaccard index to evaluate each of the method. 
# 4.Results 
From the notebook, the result is the table below:
| Algorithm          | Jaccard | F1-score | LogLoss |
|--------------------|---------|----------|---------|
| KNN                | 0.56302       | 0.547122        | NA      |
| Decision Tree      | 0.56285       | 0.534773        | NA      |
| LogisticRegression | 0.52435       | 0.509146        | 0.68563     |
# 5.Discussion
From the result above, the Jaccard index and F1-Score are close among these three method. However, the Logistic Regression has a better Log Loss number. The reason for that is the SEVERITYCODE in this data set is binary, which is only two classes: class 1 and class 2. 
# 6.Conclusion
In conclusion, the Logistic Regression is the best way to classify this dataset. Therefore, a data can be classify by given the weather, road condition and light condition. 
