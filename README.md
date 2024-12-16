Data Source and Description:
As juniors in college preparing for critical internship applications, we recognized the importance of understanding how resumes align with industry expectations. Automated resume screening has become a pivotal part of recruitment, and we were inspired to explore its inner workings. By developing our own resume classification tool, we aimed to streamline and optimize our resumes while gaining deeper insights into what makes a strong candidate in various industries. The dataset was sourced from Kaggle. It contains over 2,400+ resumes, each associated with a job category (e.g., “Information-Technology”, “HR”, “Finance”, etc.). Each record in the CSV includes:
ID: A unique identifier for each resume (also corresponds to a PDF filename).
Resume_str: The plain text version of the resume.
Resume_html: An HTML-formatted version of the resume (not used extensively in this analysis).
Category: The job category the resume falls under (e.g., HR, Finance, Engineering).

Data Cleaning and Preprocessing:
We converted all resume text to lowercase, removed stopwords, and lemmatized tokens using spaCy to standardize the textual data.
We applied a Sentence-BERT model to transform the resumes into embeddings, allowing us to measure their similarity to a user’s uploaded resume or a query prompt.
We extracted and counted industry-specific skills by defining skill dictionaries for each job category. For the chosen industry (“Information-Technology”), we counted occurrences of keywords like “python,” “sql,” “cloud,” “devops,” etc.
We extracted years of experience mentioned in the resume text using regex patterns (e.g., searching for patterns like “X years”). Based on the extracted years, we categorized candidates into levels: junior (<3 years), mid (3-7 years), senior (7-15 years), and executive (>15 years).

Data Concerns and Steps Taken:
The dataset is biased towards textual data and may not uniformly represent all categories. Some categories may have more samples than others, potentially skewing results.
We used SMOTE, an oversampling technique, to handle class imbalance when building our predictive model. This allowed the model to better learn from underrepresented classes.

Statistical Test: 
We conducted a two-sample t-test comparing the average similarity scores of resumes categorized as “Engineering” vs. those categorized as “Finance” to the user’s resume.
Null Hypothesis (H0): The mean similarity score to the user’s resume for Engineering resumes is the same as for Finance resumes.
Alternative Hypothesis (H1): The mean similarity score differs between Engineering and Finance resumes.
Result:
The p-value was effectively zero, leading us to reject H0. There is a statistically significant difference between the similarity scores of Engineering and Finance resumes relative to the user’s resume. The takeaway is that these categories differ in how closely they match the user’s profile based on the chosen query prompt.

Variable Relationships (Correlation): 
We computed a correlation between a numeric encoding of the resume categories and the user similarity scores. While we found a small positive correlation (~0.10), we cannot interpret this as a causal relationship. It merely suggests that certain numeric encodings of categories slightly correlate with higher similarity scores, but this numeric mapping was arbitrary.

Causal Interpretation: 
Causality cannot be inferred in this scenario because we have no temporal ordering, no controlled experiment, and multiple confounding factors. The correlation found is not evidence that a certain category causes higher similarity scores. There are many other variables (resume structure, skill sets, length) that influence similarity. Without a randomized controlled setup or a causal framework, we cannot claim a causal relationship.

Predictive Model: 
We built a predictive model to determine if a given resume is a “good fit” for a chosen industry (e.g., Information-Technology) and experience level (junior). The steps included:
Features:
User_Similarity: Cosine similarity between the resumes’ embedding and a query prompt embedding.
Skill_Count: Count of industry-specific skills.
Experience_Num: Numeric encoding of experience level (junior=1, mid=2, senior=3, executive=4).
Model and Handling Imbalance:
We chose a logistic regression model. To address class imbalance (the chosen industry might be underrepresented), we:
Used class_weight='balanced' in logistic regression.
Applied SMOTE to oversample the minority class in the training set.
Tuned the classification threshold to maximize F1-score for the positive class.
In and Out-of-Sample Evaluation:
We split the data into training and test sets. After training and tuning the threshold on the training set, we evaluated the final model on the test set. The AUC remained high (~0.91), and F1-score for the positive class improved significantly (from near 0.3 to about 0.59), demonstrating better in-sample and out-of-sample performance.
Interpretation:
Higher similarity scores and skill counts tend to increase the probability of predicting a “good fit.”
The experience level feature added nuance: resumes with the correct experience level category aligned better with the prompt, improving classification performance.

Data Visualization:
We included several data visualizations:
Boxplot of User Similarity by Category: This visualization displayed how different job categories varied in their similarity scores to the user’s resume.
ROC Curve: The ROC curve for the logistic regression model showed a high AUC, indicating strong discriminatory power. This was accompanied by the chosen threshold that improved F1-score for the minority class.



User Similarity Scores by Category (Boxplot)

This boxplot shows the distribution of user similarity scores across various job categories. The x-axis represents different job categories (e.g., HR, Designer, Information Technology, etc.), while the y-axis indicates the similarity scores.

Key Observations:
Central Tendency: The boxes represent the interquartile range (IQR), with the line inside each box showing the median similarity score for each category.
Spread and Outliers: The whiskers extend to show the range of scores within 1.5 times the IQR, and the dots outside the whiskers represent outliers.
Category Variance: Categories like "Designer" and "Healthcare" exhibit higher median scores and less variability compared to others like "Banking" or "Aviation."
Insights: Categories with higher similarity scores may reflect more uniform user behavior or characteristics, while those with wider variability suggest diverse patterns.

ROC Curve for Good Fit Prediction
This is a Receiver Operating Characteristic (ROC) curve used to evaluate the performance of a Logistic Regression model in predicting a binary outcome (e.g., whether a user is a "good fit").

Key Elements:
Axes:
The x-axis represents the False Positive Rate (FPR), which is the proportion of negatives incorrectly classified as positives.
The y-axis represents the True Positive Rate (TPR), which is the proportion of positives correctly classified.
Model Performance:
The blue curve shows the model's performance as the threshold for classification is varied.
The dashed diagonal line represents the performance of a random classifier (AUC = 0.5).


AUC:
The Area Under the Curve (AUC) is 0.91, indicating strong predictive performance. AUC values close to 1 suggest excellent model performance, while values near 0.5 indicate no better than random guessing.

Insights:
The high AUC value demonstrates that the logistic regression model effectively distinguishes between the two classes in the dataset. This suggests that the predictors used in the model are relevant and informative.

Summary: What We Learned 

Data Preprocessing and Feature Engineering:

Textual data is highly unstructured, requiring extensive preprocessing to derive meaningful insights.
Sentence embeddings and skill extraction are powerful tools for understanding and quantifying resume relevance to specific industries and queries.

Statistical Analysis:
Hypothesis testing revealed significant differences in how resumes from various industries align with a user’s profile, highlighting the importance of industry-specific tailoring in applications.

Class Imbalance and Predictive Modeling:
Handling class imbalance is crucial for improving model performance in underrepresented categories.
Logistic regression combined with oversampling techniques like SMOTE and proper threshold tuning can effectively balance predictive power across classes.

Model Evaluation:
AUC and F1-scores are critical metrics to evaluate model performance, especially when dealing with imbalanced datasets.
Our results demonstrated the importance of nuanced features, such as user similarity and experience levels, in improving classification outcomes.

Practical Takeaways:
Higher similarity scores and relevant skill counts significantly influence the likelihood of a resume being classified as a “good fit.”
Tailoring resumes to highlight industry-relevant skills and aligning with experience-level expectations are vital for improving job application success.

Visualization as a Tool:
Visualizations, such as boxplots and ROC curves, are not just aids in interpretation but also serve as an effective means of communicating findings and model performance.

Conclusion
This project is just the beginning of exploring how data science can optimize job application processes. It has been a compelling exercise in applying data science to solve real-world problems, combining academic knowledge with practical applications to make job searches more efficient and data-driven.

