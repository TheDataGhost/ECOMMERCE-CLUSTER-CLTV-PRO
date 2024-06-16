Project name                    :  E-commerce Customer Segmentation & Customer Lifetime Value Prediction

Algorithms and Models   :  K-means Clustering,Principal Component Analysis,Naive Bayes, Random Forest, Ada boost, KNN, Decision Tree , Gamma-Gamma and BG/NBD Models

Visualisation Tools          :  Matplotlib, Seaborn, Plotly

Platform                           :  Visual Studio code

Objectives and Scope of E-commerce Customer Segmentation and CLTV Prediction

Customer Segmentation: Objectives and Scope

Objective :  

Enhanced Marketing Strategies, Personalised Customer Experience, Resource Allocation, Product Recommendation, Behavioural Insights

Scope: 

Data Collection
Segmentation Criteria: Define segmentation criteria such as purchase frequency, average order value, geographic location, and customer interests,
Segmentation Techniques: Use clustering algorithm(e.g., K-means) to group customers into meaningful segments,
Evaluation: Evaluate the quality and relevance of segments
Implementation and Continuous Monitoring

Customer Lifetime Value (CLTV) Prediction: Objectives and Scope
Objective :

Revenue Forecasting
Customer Retention
Marketing ROI
Strategic Decision Making: Make informed decisions on pricing, promotions, and product launches based on predicted CLTV
Personalisation: Enhance personalisation strategies by understanding the long-term value of different customer segments.

Scope:

Data Collection,Feature Engineering,Model Selection,Training and Validation,Implementation,Actionable Insights: Use CLTV predictions to segment customers, tailor marketing efforts, and prioritise customer service initiatives.

Key steps: 

Data Collection: Gather data from reliable sources—transaction logs, customer profiles, product details of 1000000 customers. Ensure data is comprehensive and relevant.
Data Preprocessing: Cleanse the data by handling missing values, removing duplicates, and normalising fields. Standardising data formats is crucial for accurate analysis.
Feature Selection & Engineering: Identify key features influencing clustering. Engineer new features if necessary, like customer lifetime value or purchase frequency.
Outlier Detection & Handling: Identify and handle outliers to ensure they do not skew the clustering results.
Dimensionality Reduction : Apply techniques like PCA (Principal Component Analysis) to reduce the number of features while preserving variance.
Cluster Analysis and Profiling :Analyse and profile clusters to understand the characteristics and behaviours of each segment.
Model Training & Optimization: Train the model using selected algorithms. Optimise parameters to improve cluster quality and coherence.
Evaluation of Clusters: Evaluate the clusters through visual inspection to ensure they are meaningful and well-separated.
Interpretation & Insights: Derive actionable insights by understanding the characteristics of each cluster.
Visualization & Reporting: Visualize clusters with plots and graphs; prepare comprehensive reports for stakeholders.

Key Steps in CLTV Prediction Using Gamma-Gamma and BG/NBD Models
Objective: Predict the Customer Lifetime Value (CLTV) to inform marketing, sales, and customer retention strategies.
Scope: Focus on e-commerce customers' purchase behaviour and revenue generation over a specified future period.

Data Collection:
Transaction Data: Collect historical data on customer transactions, including purchase dates, amounts, and customer IDs.
Customer Demographics: Gather additional customer information such as demographics, if available.

Data Preprocessing:

Aggregation: Aggregate transaction data to calculate metrics like frequency (number of repeat purchases), recency (time since last purchase), and monetary value (average purchase value).

Feature Engineering:

Additional Features: Create features like customer tenure, average inter-purchase time, etc.

Model Selection:
Fit BG/NBD Model: Use the frequency and recency data to fit the BG/NBD model. This model predicts the probability of future transactions based on past behaviour.
Fit Gamma-Gamma Model: Apply the Gamma-Gamma model to predict the average transaction value, using the monetary value data.
Combine these to calculate the CLTV:Expected Number of Transactions×Expected Monetary Value

Interpretation & Insights:
Customer Segmentation: Segment customers based on predicted CLTV to identify high-value and low-value segments.
Behavioural Patterns: Analyse the behavioural patterns of different customer segments to understand key drivers of value.

Visualization & Reporting:

Visualization: Create visualizations such as histograms, scatter plots, and bar charts to present the distribution of CLTV, frequency, recency, and monetary value.
Reporting: Prepare comprehensive reports to communicate findings to stakeholders, highlighting actionable insights and strategic recommendations.

Implementation:

Integration: Integrate the CLTV predictions into the company's CRM and marketing systems to enable targeted marketing and personalised customer interactions.
Automation: Set up automated processes for periodic data collection, model retraining, and CLTV prediction updates.


Key steps: Advanced E-commerce Customer Segmentation

Data Preprocessing:
Data Cleaning: Handle missing values, remove duplicates, and standardise data formats.
Feature Engineering: Aggregate data to calculate frequency, recency, average monetary value, and CLTV for each customer.

Feature Selection:
Key Features: Select important features for segmentation such as frequency, recency, average monetary value, and CLTV.

Model Selection and Training:
Algorithm Choice: Applied Algorithms Like Naive Bayes, Random Forest, Ada boost, Ken, Decision Trees to the preprocessed data

Imbalanced Data Handling:
Oversampling: Use techniques like SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset by increasing the number of instances in underrepresented segments.
Under sampling: Apply random under sampling to reduce the number of instances in overrepresented segments, ensuring balanced data distribution.

Hyperparameter Tuning:
Tuning Algorithms: Use Grid Search or Random Search to optimise hyper parameters of clustering algorithms for improved segmentation accuracy.

Model Training & Evaluation:
Train Model: Train the clustering model on the preprocessed and balanced dataset.
Evaluate Model: Evaluate the quality of segment using metrics like classification report or confusion metrics.
           Model Accuracy: Achieved an impressive 94.4% accuracy in model prediction, indicating     high reliability and precision in customer segmentation.

Interpretation & Insights:
Analyse Segments: Examine the characteristics and behaviours of each customer segment.
Actionable Insights: Derive actionable insights to inform marketing strategies, product development, and customer engagement plans.

created a front-end design in Streamlit to visualize and interact with the customer segmentation data.

