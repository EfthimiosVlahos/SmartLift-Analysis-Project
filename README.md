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

---


# Data Processing <a id="part-2"></a>

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


## Preparing Dataset

By loading individual accelerometer and gyroscope CSV files located in the MetaMotion directory. Every CSV file present in this directory is cataloged for easy referencing. From each file's name, vital metadata such as the participant's identity, exercise label, and the activity's category are extrapolated. 

Utilizing this mined information, we initialize dataframes specifically for the accelerometer and gyroscope data. As we traverse through the list of files, the data is read, categorized based on its source (either accelerometer or gyroscope), and then amalgamated into the respective dataframe. To optimize the data processing workflow, we've encapsulated the logic for parsing and categorizing the data within the `read_data_from_files` function. This function not only amalgamates the data but also manages timestamp conversions and eradicates redundant columns. 

Post the utilization of this function on all files, we combine the accelerometer and gyroscope datasets. To maintain data uniformity and facilitate its manageability, the consolidated data undergoes a resampling process based on designated frequency parameters. Concluding the data processing phase, the cleansed and structured data is archived into a new file, rendering it primed for the subsequent stages of analysis.


<table>
  <tr>
    <td><img src="https://github.com/EfthimiosVlahos/SmartLift-Analysis-Project/assets/56899588/d2e278cd-811b-4828-8afb-15e551497641" height="400" width="400"></td>
    <td><img src="https://github.com/EfthimiosVlahos/SmartLift-Analysis-Project/assets/56899588/3d62c5cc-6e74-44a3-a82f-e29bbb963555" height="400" width="400"></td>
  </tr>
  <tr>
    <td colspan="2" style="text-align:center">Figure 1: Preparing Dataset</td>
  </tr>
</table>




## Converting Raw Data

The raw dataset comprised 69,677 entries, each consisting of a timestamp and x, y, z sensor readings. Given the distinct timestamps for each sensor reading, data needed aggregation. An aggregation step size of Δt = 0.20 s (or five readings per second) was chosen, with numerical values aggregated using the mean and categorical attributes (labels) using the mode.

Two primary strategies were employed to manage the raw data: Low-pass Filtering for individual attributes and Principal Component Analysis (PCA) for the entire dataset.


## EDA

In this project, I utilized Python's renowned data manipulation and visualization libraries, pandas and Matplotlib, to probe into accelerometer and gyroscope data. My exploration started by visualizing the 'acc_y' column data for the initial set of exercises. To grasp the distinct characteristics of various exercises, I navigated through the unique exercise labels and plotted the 'acc_y' column for each. This approach showcased both the full dataset and a truncated first 100 rows, offering not just a comparative view of exercises but also an immediate glimpse of early data patterns.

Delving deeper, I fine-tuned the visual settings using Matplotlib to meet my analytical requirements. I then compared the 'acc_y' data between medium vs. heavy sets and across different participants for chosen exercises. This exploration unveiled a holistic perspective as I plotted all the accelerometer axes concurrently, leading to a thorough comprehension of the exercise data.

Broadening the scope, I cycled through all the combinations of exercises and participants. This approach provided a detailed and overarching view of the accelerometer and gyroscope data. This depth of analysis was pivotal when trying to discern the subtle differences and parallels between participants for each exercise. Concluding the visualization, I merged accelerometer and gyroscope plots into a unified figure. I also journeyed through all exercise and participant combinations, devising combined plots for both sensors. These intricate visualizations were subsequently saved for further analysis and reference.

![acc_y_med_v_heavy](https://github.com/EfthimiosVlahos/SmartLift-Analysis-Project/assets/56899588/aa6e437b-1c15-4c89-a63f-d62cf988d08a)
*Figure: 'acc_y' data between medium vs. heavy sets and across different participants for chosen exercises*

![acc_y_data_across_part](https://github.com/EfthimiosVlahos/SmartLift-Analysis-Project/assets/56899588/760ec5ce-2de1-4a64-881e-d4fbf701dd32)
*Figure: 'acc_y' across different participants for a specific exercise label*

![acc_squat](https://github.com/EfthimiosVlahos/SmartLift-Analysis-Project/assets/56899588/b18cc1f2-bcee-4b13-a090-3638255dbe9f)
*Figure: Plot all accelerometer axis data for a specific participant and exercise label (squat)*

![acc_gyr_plot](https://github.com/EfthimiosVlahos/SmartLift-Analysis-Project/assets/56899588/b5e00eea-0d2f-4bd6-af51-0b4ea5a74154)
*Figure: Accelerometer and gyroscope plots in a single figure for a specific participant and exercise label*


# Detecting Outliers <a id="part-3"></a>

Within the SmartLift Analysis, ensuring the quality of the data is paramount. This document provides a deep dive into the methodologies implemented for outlier detection, crucial to data integrity and the subsequent analysis.

## Visualization of Outliers

Data visualization plays a pivotal role in the early identification of outliers. Utilizing the `matplotlib` library, the data was visualized using boxplots, allowing for a quick assessment of data spread and potential outliers. Specific columns such as `'acc_x'` and `'gyr_y'` were singled out for detailed visual representation.


<table>
  <tr>
    <td>
        <img src="https://github.com/EfthimiosVlahos/SmartLift-Analysis-Project/assets/56899588/0e38d223-5800-4324-b459-4a91fb7e77c0" alt="acc_x_groupby_label"/>
    </td>
    <td>
        <img src="https://github.com/EfthimiosVlahos/SmartLift-Analysis-Project/assets/56899588/5498aaf7-9974-4059-997a-88dd8efbb818" alt="gyr_y_groupby_label"/>
    </td>
  </tr>
  <tr>
    <td>
        acc_x grouped by label
    </td>
    <td>
        gyr_y grouped by label
    </td>
  </tr>
</table>




## Interquartile Range (IQR)

The IQR method, widely recognized for its efficacy, identifies outliers based on quartiles of the dataset:
1. First (Q1) and third (Q3) quartiles are computed.
2. IQR is derived as the difference between Q3 and Q1.
3. Lower and upper bounds are calculated as: 
   - `Lower Bound = Q1 - 1.5 x IQR`
   - `Upper Bound = Q3 + 1.5 x IQR`
4. Data points falling outside these bounds are deemed outliers.

The function `mark_outliers_iqr` has been employed to mark outliers using the IQR method.

### Chauvenet's Criterion

Chauvenet's criterion assumes a normal distribution for data. For each data point:
1. The probability of observing its value, given the mean and standard deviation, is ascertained.
2. This probability is benchmarked against a specific criterion value; values falling below this threshold are tagged as outliers.

Each column in the dataset was assessed for outliers using Chauvenet's criterion. The visualization is achieved through the `plot_binary_outliers` function.

### Local Outlier Factor (LOF)

LOF is an outlier detection method grounded on the concept of data density. It assesses the local density deviation of a data point compared to its neighbors. Data points with a significantly lower density than their neighbors are treated as outliers.

The `mark_outliers_lof` function has been used to detect and mark outliers based on LOF.


<table>
  <tr>
    <td>
        <img src="https://github.com/EfthimiosVlahos/SmartLift-Analysis-Project/assets/56899588/3763e81d-ed19-45ac-9dfb-b5bd6e77f586" alt="Chauvanet" height="400" width="400"/>
    </td>
    <td>
        <img src="https://github.com/EfthimiosVlahos/SmartLift-Analysis-Project/assets/56899588/29b8a40f-2cb1-41a8-a41d-bdaf6a3f8efe" alt="LOF" height="400" width="400"/>
    </td>
  </tr>
  <tr>
    <td>
        Chauvanet Method
    </td>
    <td>
        LOF Method
    </td>
  </tr>
</table>





### Outlier Treatment

Post identification, outliers are addressed by replacing them with NaN values, ensuring these anomalous values do not adversely impact further data analyses. 

### Exporting the Processed Data

Following outlier detection and treatment, the cleaned dataset has been saved and is ready for subsequent processing phases. For access to the processed dataset, refer to the file: `02_outliers_removed_chauvenets.pkl`.


<table>
  <tr>
    <td>
        <img src="https://github.com/EfthimiosVlahos/SmartLift-Analysis-Project/assets/56899588/979b9c94-bab5-4571-965a-774a640e0153" alt="Mark_outliers" height="300"/>
    </td>
    <td>
        <img src="https://github.com/EfthimiosVlahos/SmartLift-Analysis-Project/assets/56899588/ee3b4efe-5d85-4cfc-b59b-0e521eda5bf9" alt="NAN" height="300"/>
    </td>
  </tr>
  <tr>
    <td>
       Marking Outliers
    </td>
    <td>
        Replacing Outliers with NAN
    </td>
  </tr>
</table>



---


# Feature Engineering <a id="part-4"></a>

## Data Loading and Initial Exploration
I begin by loading the dataset, which has already been preprocessed to remove outliers based on Chauvenet's criterion. A cursory inspection of the data structure reveals that our primary focus will be on the first six columns, which serve as predictor columns. Some basic visualization techniques, like plotting values from the 'gyr_y' column, aid in initial data understanding. Data often has missing values, which can adversely impact many machine learning algorithms. As a part of our preprocessing pipeline, we've used interpolation to fill in gaps in our data series, specifically in the predictor columns. The built-in `interpolate()` method in Pandas provides a quick and effective way to address this. To understand the data's temporal dimension, we've calculated the duration for each unique set within our dataset. Further insights were gained by plotting values from the 'acc_y' column for specific sets and computing mean durations across different categories.


## Feature Engineering: Advanced Transformation and Modeling

<table>
  <tr>
    <td>
      <img src="https://github.com/EfthimiosVlahos/SmartLift-Analysis-Project/assets/56899588/32b704a9-3658-414a-a079-76b2db6c0308" width="400" height="400"/>
    </td>
    <td>
      <img src="https://github.com/EfthimiosVlahos/SmartLift-Analysis-Project/assets/56899588/66cec8c3-a7ca-4b1e-aeb7-dbee3f5a799f" width="400" height="400"/>
    </td>
  </tr>
  <tr>
    <td align="center">Interpolation</td>
    <td align="center">Calculating Duration</td>
  </tr>
</table>

## Low-Pass Filtering with Butterworth
Signal processing is a key part of our feature engineering process. The Butterworth low-pass filter has been employed to reduce high-frequency noise in our data. This step is crucial for highlighting the fundamental patterns in our signals, which would be the focus for any downstream modeling process. The low-pass filter is applied to each of the predictor columns, thereby replacing the original values with their filtered counterparts.
## Data Preprocessing: Filtering Techniques

<table>
  <tr>
    <td>
      <img src="https://github.com/EfthimiosVlahos/SmartLift-Analysis-Project/assets/56899588/a9a83d5b-c996-4b93-a913-bd92478693b2" width="500" height="400"/>
    </td>
    <td>
      <img src="https://github.com/EfthimiosVlahos/SmartLift-Analysis-Project/assets/56899588/601eb832-8acd-4a22-bed1-55545be7e06b" width="500" height="400"/>
    </td>
  </tr>
  <tr>
    <td align="center">Low-Pass Code</td>
    <td align="center"> Compare with and without Low-Pass filter</td>
  </tr>
</table>

## Principal Component Analysis (PCA)
High dimensionality can often be a challenge in machine learning. PCA aids in dimensionality reduction by transforming the original predictor columns into a set of orthogonal components that capture the most variance. Our analysis indicated that we can effectively capture a significant portion of the variance in the data using just the first three principal components.

<table>
    <tr>
        <td>
            <img src="https://github.com/EfthimiosVlahos/SmartLift-Analysis-Project/assets/56899588/c70e1f24-b97d-465f-8f32-aa463a4311d1" alt="f_3_pca" width="500"/>
        </td>
        <td>
            <img src="https://github.com/EfthimiosVlahos/SmartLift-Analysis-Project/assets/56899588/be59b002-abac-451a-a6b4-fc7ffd44f9dd" alt="expl_var" width="500"/>
        </td>
    </tr>
    <tr>
        <td style="text-align:center; padding-top:10px;">
            Explained variance for each principal component
        </td>
        <td style="text-align:center; padding-top:10px;">
            Elbow Method for PCA
        </td>
    </tr>
</table>





## Sum of Squares Attributes & Temporal Abstraction
Following the dimensionality reduction achieved through Principal Component Analysis (PCA), we derived squared magnitudes for accelerometer and gyroscope readings to obtain a singular representation of sensor activity intensity. With these magnitudes in hand, a temporal abstraction was performed. This utilized a window-based strategy to segment the data temporally, providing both mean and standard deviation metrics, ultimately enhancing our understanding of the temporal fluctuations in the sensor data.

## Frequency Features & Overlapping Windows Resolution
Venturing into the frequency domain, we employed Fourier Transformations to extract critical frequency domain attributes such as maximum frequency, frequency weighting, and power spectral entropy. In the wake of these transformations, the challenge of overlapping windows emerged. This was systematically addressed by filtering out rows with `NA` values and strategically skipping every alternate row to ensure non-overlapping, consistent data segments.

## Clustering Analysis
With our data processed and refined, it was primed for clustering analysis. Leveraging the k-means clustering algorithm, we assessed a range of potential cluster numbers to discern the optimal cluster count, achieved by examining the sum of squared distances or inertia against each potential cluster number. A 3D scatter plot offered a vivid visualization of the resultant clusters, centered predominantly around accelerometer readings. Further, a distinct visualization based on data labels underscored the efficacy of the clustering. The concluding step in our pipeline involved serializing and exporting the fully processed dataset, paving the way for future analyses or potential deployment scenarios.



![clustering](https://github.com/EfthimiosVlahos/SmartLift-Analysis-Project/assets/56899588/db0069ef-7bba-4281-9233-f8eac8bc183d)
*Fig. 1. Clustering based off label*




# Predictive Modeling <a id="part-5"></a>

In the predictive modeling and rep counting section, we begin with the necessary data preparations. After importing the essential libraries, the dataset is loaded from the `03_data_features.pkl` file. We drop non-relevant columns such as `participant`, `category`, and `set`. The dataset is then divided into features (X) and the target label (y). We utilize a 75-25 train-test split, ensuring an even distribution of the target variable `label` across both subsets. To visualize the distribution of our target variable, we plot the `label` distribution for the full, training, and test sets.

Following data preparation, we delve into feature engineering. Features are categorized based on their characteristics, such as Basic Features (most likely derived from accelerometer and gyroscope data), Squared Features (which represent magnitudes), PCA Features, Time-Related Features, Frequency-Related Features, and Cluster Features. Using these categories, we form four distinct combinations of feature sets for model training.

The next step involves forward feature selection using a simple decision tree. This iterative process starts with no features and successively adds features that optimize model accuracy. A visual representation illustrates the progression of accuracy as more features are incrementally included.

Lastly, we embark on the model training and evaluation phase. Using the various feature subsets, several models are trained, namely Neural Network, Random Forest, Decision Tree, and Naive Bayes. Each model's performance is gauged using accuracy and stored in a DataFrame. Notably, non-deterministic classifiers like Neural Networks and Random Forests undergo multiple training iterations to provide a more stable performance estimate. The culmination of this section aims to identify the optimal feature-model combination that best encapsulates the inherent patterns in the data, facilitating effective predictive modeling and rep counting.



<table>
    <tr>
        <td>
            <img src="https://github.com/EfthimiosVlahos/SmartLift-Analysis-Project/assets/56899588/ffee6b7f-6e0f-4cf8-b259-d1759188ebbd" alt="Feature Splitting" >
            <p align="center"><b>Feature Splitting</b></p>
        </td>
        <td>
            <img src="https://github.com/EfthimiosVlahos/SmartLift-Analysis-Project/assets/56899588/8716cc15-8065-4f37-bab2-0b9a0ae4e0b5" alt="Grid Search and Model Performance">
            <p align="center"><b>Grid Search and Model Performance</b></p>
        </td>
    </tr>
</table>



## Models 

We kick off this section by plotting a grouped bar plot to visually compare the accuracy of different models with varied feature sets. After sorting our score DataFrame based on descending accuracy values, we employ Seaborn's `barplot` to render this visualization. This plot is instrumental in distinguishing the top-performing model-feature set combinations at a glance.

Moving on, the spotlight is cast upon the best performing model: the Random Forest classifier. It's trained and evaluated on `feature_set_4`. Post-training, we leverage a confusion matrix to gauge the model's performance intricacies. This matrix elucidates the true positives, true negatives, false positives, and false negatives, providing an in-depth perspective on classification results.

Next, our focus shifts to participant-based data segregation. We curate our training and testing sets based on a particular participant, labeled "A". After filtering, the `participant` column is removed to avoid redundancy. We then visualize the distribution of labels within these newly formed datasets. The distribution ensures we have a comprehensive understanding of the class balance, or potential imbalance, between our training and testing sets.

Having our data stratified on the participant, the best model (Random Forest) is put to the test again. As before, post-training, we employ a confusion matrix to shed light on its classification nuances. This matrix is particularly useful in observing any performance disparities when models are trained and tested on data segregated based on individual participants.

Finally, in our quest for model simplicity without compromising efficacy, we venture into training a feed-forward neural network using a select set of features. The aim is to discern if a simpler model can rival, or potentially outdo, the performance of the more complex Random Forest. As is consistent with our prior evaluations, a confusion matrix follows to illustrate the neural network's classification nuances, providing a holistic understanding of its prediction prowess.


# Repetition Counting <a id="part-7"></a>

Our journey commences by setting the stage with the necessary libraries. Apart from the staple ones like `numpy`, `pandas`, and `matplotlib`, we also rope in `LowPassFilter` from `DataTransformation`. This module is particularly pivotal in noise reduction from our signals.

The default Pandas chained assignment warning is silenced, ensuring that the exploratory data analysis remains unhindered by unwanted warnings.

Our plotting preferences lean towards the "fivethirtyeight" style, often regarded for its visually pleasing aesthetics. With figure size and DPI set, our plots promise clarity and precision.

## Data Loading and Feature Engineering

We initiate by loading the processed data. Here, records with the label "rest" are excluded, perhaps owing to their irrelevance for the current analysis. Features capturing the magnitude of acceleration (`acc_r`) and gyroscope (`gyr_r`) are then calculated, which can be particularly insightful when distinguishing movements based on their intensities.

## Dataset Segmentation

For ease of analysis, our primary dataset is bifurcated into five subsets based on distinct exercise labels. This enables us to drill deeper into patterns unique to each exercise.

## Data Visualization

To discern underlying patterns, a subset of the bench press data is visualized. The aim is to juxtapose readings from the accelerometer's three axes with their resultant magnitude. Similar visual insights are derived for gyroscope readings.

## Noise Reduction with LowPassFilter

Signal noise can be a significant impediment, often masquerading genuine patterns. Hence, a LowPass filter, which attenuates high-frequency noise, is applied to our dataset. It's particularly enlightening to visualize how the original signal differs post this filtration.

### Repetition Counting

With noise filtered, we embark on the crucial task of repetition counting. The function `count_reps` aids in counting peaks, which essentially translate to repetitions in our exercises. Visual aids, like marking these peaks, ensure that the counting logic is verifiable.

### Benchmarking and Repetition Prediction

Our dataset is then enriched with a 'reps' column, where the number of repetitions is benchmarked based on the exercise category. Following this, a DataFrame, `rep_df`, is curated, which aims to compare actual repetitions with predicted ones. This comparison is achieved by employing the earlier defined `count_reps` function.

### Evaluation

Lastly, it's paramount to evaluate the efficacy of our repetition prediction approach. The mean absolute error between actual and predicted repetitions serves as our evaluation metric. A bar plot subsequently offers a visual comparison between these values, segmented by exercise and category.




# Conclusion <a id="part-7"></a>

Over the course of this project, we delved deep into the realms of biomechanical data processing, feature engineering, and predictive modeling to address key challenges in strength training analytics.

1. **Rich Feature Engineering:** Our comprehensive approach to extracting features from raw accelerometer and gyroscope readings, leveraging techniques from aggregation, time-domain analysis, frequency-domain (Fourier Transformation), and clustering, unveiled novel patterns and insights from the data, setting a solid foundation for subsequent modeling.

2. **Classification Excellence:** Through meticulous model selection and tuning, our Random Forest model stood out, achieving an astounding accuracy of 98.51% for exercise classification, solidifying the effectiveness of our feature engineering endeavors.

3. **Repetition Counting:** The peak counting algorithm, tailored to specific exercises, demonstrated its prowess with a decent accuracy, with the most notable success in deadlifts and overhead presses.

4. **Form Detection:** The ability to discern improper form in exercises like the bench press with a staggering accuracy of 98.53% underscores the potential of our approach in ensuring safe and effective strength training. Such analytics could be invaluable for trainers and trainees alike, emphasizing proper form for optimal gains and injury prevention.

5. **Generalization Insights:** While our models showcased significant prowess, the variations in accuracy based on different weight classes and participants underscore the ever-present challenges in model generalization. Yet, the attained average accuracy of 85.43% across participants remains commendable.

Moving forward, the techniques and insights gleaned from this project bear the potential to revolutionize the domain of strength training analytics. By combining raw biomechanical data with sophisticated algorithms, we pave the way for more informed, safe, and effective training regimens. The promise of real-time feedback on form and counting, backed by such robust analytics, could truly elevate the training experience for many.


