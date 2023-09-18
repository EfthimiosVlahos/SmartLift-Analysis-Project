# Smart Lift Project

## Table of Contents
- [Introduction, Goal, and Context](#part-1)
- [Data Processing](#part-2)
- [Outlier Handling](#part-3)
- [Feature Engineering](#part-4)
- [Predictive Modelling and Repetition Counting](#part-5)
- [Conclusion](#part-6)

# Introduction <a id="part-1"></a>

In the past decade, breakthroughs in sensor technology have made wearable devices like accelerometers, gyroscopes, and GPS-receivers more feasible and accessible. Such advancements have propelled the monitoring and classification of human activities to the forefront of pattern recognition and machine learning research. This is majorly due to the immense commercial potential of context-aware applications and evolving user interfaces. Beyond commerce, there's a broader societal impact: addressing challenges related to rehabilitation, sustainability, elderly care, and health.

Historically, the focus was largely on tracking aerobic exercises. Systems existed to monitor running pace, track exertion, and even automate some functionalities of exercise machines. However, the domain of free weight exercises remained relatively uncharted. There's a notable gap: while aerobic exercises have been well-addressed by wearables, strength training – a crucial component of a balanced fitness regime – hasn't been explored to its full potential.

Digital personal trainers might soon be a reality, with advancements in context-aware applications. While there have been significant strides towards this future, there remains a vital component yet unaddressed: tracking workouts effectively and ensuring safety and progress.

This project is centered around this very aspect: exploring possibilities within the strength training domain by leveraging wristband accelerometer and gyroscope data. This data, collected during free weight workouts from five participants, serves as the foundation. The overarching goal? Build models akin to human personal trainers that can track exercises, count repetitions, and detect improper form.

Stay tuned as we traverse through related works, delve into the data collection method, process raw data, engineer features, build predictive models, and draw conclusions.

---


# Part 2 — Data Processing <a id="part-2"></a>

## Introduction to Data Consideration:

Other works have demonstrated the application of various machine learning algorithms to free weight exercise accelerometer data, with significant outcomes. From a strength training standpoint, there seems to be an oversight in collecting quality datasets. This study addresses this gap by creating an experimental environment that replicates real strength training workouts.

Many related studies appear to select random sets of exercises without clear motivations for their choices. While capturing a broad range of exercises is vital, this research narrows its focus to the exercises from the "Starting Strength" training program by Mark Rippetoe. This decision aims to avoid noise from exercise combinations that wouldn't typically be performed together in genuine workout scenarios. The "Starting Strength" program encompasses five primary barbell exercises: Bench Press, Deadlift, Overhead Press, Row, and Squat, as visualized in Figure 1.

![Basic Barbell Exercises](path_to_image)  
*Fig. 1. Basic Barbell Exercises*

Strength training programs often use terms like sets and repetitions (reps). Here, a rep represents a single completion of an exercise, while a set is a sequence of reps followed by a rest period. "Starting Strength" emphasizes using heavier weights, ideally chosen to allow around five reps per set. The research explores if a trained model can still accurately classify exercises considering these variations and also delves into possible methods to verify exercise form.

## Data Collection

Instead of solely relying on accelerometer data, like many previous studies, this research harnesses the capabilities of more modern smart devices. These devices, such as smartwatches, come equipped with additional sensors, including gyroscopes. For this study, data was gathered using MbientLab’s wristband sensor research kit, a device that simulates a smartwatch's placement and orientation. The data capture rates were set at 12.500Hz for the accelerometer and 25.000Hz for the gyroscope. Five participants (outlined in Table 1) executed the barbell exercises in both 3 sets of 5 reps and 3 sets of 10 reps. The higher rep sets aim to observe model generalization across different weights, amounting to a total of 150 sets of data. Additionally, 'resting' data was captured between sets without imposing any specific restrictions on the participants.

| Participant | Gender | Age | Weight (Kg) | Height (cm) | Experience (years) |
|-------------|--------|-----|-------------|-------------|--------------------|
| A           | Male   | 23  | 95          | 194         | 5+                 |
| B           | Male   | 24  | 76          | 183         | 5+                 |
| C           | Male   | 16  | 65          | 181         | <1                 |
| D           | Male   | 21  | 85          | 197         | 3                  |
| E           | Female | 20  | 58          | 165         | 1                  |

*Table 1. Participants (N=5)*

## Weights

To pinpoint the ideal weights for the exercises, the study employed the one rep max (1RM) metric. The 1RM signifies the maximum weight someone can lift in a single repetition. There are various methods to calculate 1RM, with this study utilizing Epley’s formula for its calculations.

## 3.3 Execution Form

To gauge the quality of exercise execution, additional data was obtained, particularly for the bench press. Participants were instructed to deliberately perform the exercise with errors, such as lowering the bar too high on the chest and not touching the chest altogether.

## Converting Raw Data

The raw dataset comprised 69,677 entries, each consisting of a timestamp and x, y, z sensor readings. Given the distinct timestamps for each sensor reading, data needed aggregation. An aggregation step size of Δt = 0.20 s (or five readings per second) was chosen, with numerical values aggregated using the mean and categorical attributes (labels) using the mode.

Two primary strategies were employed to manage the raw data: Low-pass Filtering for individual attributes and Principal Component Analysis (PCA) for the entire dataset.

## 4.1 Low-pass Filter

The low-pass filter is ideal for temporal data with periodicity. The Butterworth low-pass filter was used to eliminate high-frequency noise potentially affecting the learning process. After thorough inspection, a cut-off point of 1.3 Hz was selected.

## 4.2 Principal Component Analysis

A PCA was performed to discern features that explained the majority of variance in the dataset. The results illustrated a rapid decline in explained variance after the first three components. Thus, the study incorporated these three components into the dataset.


# Detecting Outliers <a id="part-3"></a>

##  Introduction to Outliers
Outliers are extreme values that deviate significantly from the other observations in the dataset. These outliers can be genuine observations or result from errors. In this project, we focus on identifying and handling these extreme values to ensure the robustness of our analysis.

## Boxplots and Interquartile Range (IQR)
Boxplots were used to visualize potential outliers across different sensor readings. The IQR method was employed to determine outliers, marking values outside 1.5 times the IQR as outliers.

## Plotting Outliers Over Time
Outliers were visualized over time using a custom function. This allowed for easy identification of patterns or anomalies within the data.

## Distribution-based Outlier Detection: Chauvenet’s Criterion
The data distribution was first visualized using histograms. Chauvenet's criterion, which assumes a normal distribution, was then applied to mark outliers.

## Distance-based Outlier Detection: Local Outlier Factor (LOF)
LOF was employed to detect outliers by comparing the density deviation of an instance with that of its neighbors.

## Outliers Grouped by Labels
To understand the impact of outliers within different labels or categories, outliers were detected and visualized for each label separately.

## Handling and Replacing Outliers
After selecting Chauvenet's criterion as the preferred outlier detection method, the outliers were replaced with NaN values to maintain the dataset's integrity without the influence of these extreme values.

## Exporting the Processed Dataframe
The dataframe, with outliers removed, was saved to an external pickle file for further analysis.

# Feature Engineering <a id="part-4"></a>

In this section, we delve into the derivation of additional features from the original dataset. These enhancements span various dimensions including aggregated, time-related, frequency, and clustering-based features.

## Aggregated Features

Leveraging the data further, the scalar magnitudes \( r \) of both the accelerometer and gyroscope were derived. The formula for \( r \) is:

\[ r_{\text{magnitude}} = \sqrt{x^2 + y^2 + z^2} \]

The strength of using \( r \) over any specific data direction lies in its impartiality towards device orientation. It provides a robust measurement even when the device undergoes dynamic re-orientations.

## Time Domain

The temporal nature of the dataset was exploited by aggregating numerical data points using standard deviation (sd) and mean. The sd aimed to capture data variations over time, anticipating higher values during exercises and lower ones during rest. Temporal means, meanwhile, shed light on the general levels of data, reducing the noise's influence. Following experimentation with various window sizes (2, 4, and 6 seconds), a 4-second window was finalized for the dataset as seen in Figure 5.

![Numerical temporal aggregation with window sizes of 2, 4, and 6 seconds](path_to_figure_5_image)

## Frequency Domain: Fourier Transformation

The frequency domain was another area of focus. With the help of a Fourier transformation, measurements were represented as combinations of sinusoid functions of varied frequencies. Using the previously selected 4-second window, frequency features like the maximum frequency, frequency signal weighted average, and the power spectral entropy were computed.

## New dataset

Post feature engineering, the dataset was enriched with various new attributes. To avoid redundancy from overlapping time windows, instances that did not meet a maximum overlap criterion were pruned. A 50% overlap was deemed acceptable, leading to a dataset trimmed down to 4,505 instances. This method proved effective against overfitting, despite a slight loss in information.

## Clustering

Exploring the potential of cluster memberships aiding label predictions, acceleration data became the focal point, given that gyroscope data yielded limited value. The k-means clustering method, with \( k=4 \), stood out for its superior silhouette score of 0.6478. The choice of 4 clusters aimed to represent different labels optimally, evident from the distribution in Figure 6 and Table 2. 

![Clusters Visualization](path_to_figure_6_image)

**Table 2: Cluster Coverage**

| Label      | Cluster 1  | Cluster 2 | Cluster 3  | Cluster 4  |
|------------|:----------:|:---------:|:----------:|:----------:|
| BenchPress |  99.88%    |  0.12%    |  0.00%     |  0.00%     |
| Deadlift   |  0.00%     |  0.00%    |  100.00%   |  0.00%     |
| OHP        |  99.28%    |  0.72%    |  0.00%     |  0.00%     |
| Row        |  0.00%     |  0.00%    |  100.00%   |  0.00%     |
| Squat      |  2.98%     |  97.02%   |  0.00%     |  0.00%     |
| Rest       |  4.14%     |  3.78%    |  50.45%    |  41.62%    |



# Predictive Modeling and Counting Reps <a id="part-5"></a>

With the dataset fully processed, it now encompasses 6 basic features, 2 scalar magnitude features, 3 PCA features, 16 time-related features, 12 frequency features, and 1 cluster feature. This section elucidates the methodology and outcomes of constructing models for classification, repetition counting, and form detection.

## Classification

Given the temporal nature of our dataset, the training and test sets were partitioned based on exercise sets. To ensure robustness, the models were trained on initial sets of each exercise, weight, and participant combination and tested on subsequent ones.

* **Feature Selection:** Forward feature selection identified the most impactful features. By incrementally adding the best features to a basic decision tree, it became evident that performance plateaued beyond 15 features. The top 5 features in terms of predictive power were: pca 1, acc y, pca 3, gyr x temp std ws 4, and acc r pse.

* **Regularization:** A regularizer was introduced to penalize complexity in models. Figure 7 reveals the nuanced relationship between regularization parameters and model accuracy.

* **Models:** An ensemble of models (Neural Network, Random Forest, Support Vector Machine, K-nearest Neighbours, Decision Tree, and Naive Bayes) underwent testing. A comprehensive grid search was executed for all models.

![Impact of Regularization on Performance](path_to_figure_7_image)

## Classification Results

Figure 8 displays the performance of each model, color-coded by the feature sets described in [2]. With these insights, the Random Forest model was finetuned using the 15 most influential features. Optimal parameters discerned through 5-fold cross validation and grid search were: minimum samples per leaf - 2, n-estimators - 100, and criterion - gini. The Random Forest model boasts an impressive accuracy of 98.51% on the test set, as visualized in the accompanying confusion matrix.

![Random Forest Performance and Classification Confusion Matrix](path_to_figure_8_image)

## Counting Repetitions

Repetition counting hinged on a peak counting algorithm employed on scalar magnitude acceleration data. With a low-pass filter (cut-off at 0.4 Hz) weeding out minor local peaks, individual exercise adjustments enhanced accuracy. Particularly, the deadlift and overhead press benefited from counting troughs, or minimum values. On the whole, this counting strategy had a modest 5% error rate across the dataset, with an exemplar of 10 deadlift repetitions illustrated in Figure 9.

![Counting Deadlift Repetitions Visualization](path_to_figure_9_image)

## Detecting Improper Form

Further experiments captured data from a participant deliberately performing the bench press with erroneous form, such as misplacing the bar on the chest or avoiding contact altogether. A Random Forest model, akin to the classification one, was trained to discern form accuracy. Three labels were used: correct form, too high, and no touch. With a dataset of 1098 instances, the model exhibited an accuracy of 98.53% on the test set.

## Generalization

Model generalization remains paramount. Initial insights on weight classes, as presented in Figure 3 from section 4, hinted at discrepancies in pace between weight classes. Testing models trained on heavy weight data on medium weight sets witnessed a decline in accuracy to 79.97%, with the reverse scenario yielding a comparable 79.51%. Evaluating across participants using a leave-one-out approach yielded an average accuracy of 85.43%.


# Conclusion <a id="part-6"></a>

Over the course of this project, we delved deep into the realms of biomechanical data processing, feature engineering, and predictive modeling to address key challenges in strength training analytics.

1. **Rich Feature Engineering:** Our comprehensive approach to extracting features from raw accelerometer and gyroscope readings, leveraging techniques from aggregation, time-domain analysis, frequency-domain (Fourier Transformation), and clustering, unveiled novel patterns and insights from the data, setting a solid foundation for subsequent modeling.

2. **Classification Excellence:** Through meticulous model selection and tuning, our Random Forest model stood out, achieving an astounding accuracy of 98.51% for exercise classification, solidifying the effectiveness of our feature engineering endeavors.

3. **Repetition Counting:** The peak counting algorithm, tailored to specific exercises, demonstrated its prowess with a decent accuracy, with the most notable success in deadlifts and overhead presses.

4. **Form Detection:** The ability to discern improper form in exercises like the bench press with a staggering accuracy of 98.53% underscores the potential of our approach in ensuring safe and effective strength training. Such analytics could be invaluable for trainers and trainees alike, emphasizing proper form for optimal gains and injury prevention.

5. **Generalization Insights:** While our models showcased significant prowess, the variations in accuracy based on different weight classes and participants underscore the ever-present challenges in model generalization. Yet, the attained average accuracy of 85.43% across participants remains commendable.

Moving forward, the techniques and insights gleaned from this project bear the potential to revolutionize the domain of strength training analytics. By combining raw biomechanical data with sophisticated algorithms, we pave the way for more informed, safe, and effective training regimens. The promise of real-time feedback on form and counting, backed by such robust analytics, could truly elevate the training experience for many.


