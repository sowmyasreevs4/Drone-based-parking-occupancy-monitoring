# Student Dropout Risk Prediction using Apache Spark & EdNet KT3

## Project Overview

This project was developed for the **CS6502 – Applied Big Data and Visualization** module at the **University of Limerick**. The goal is to build a scalable big data and machine learning pipeline using **Apache Spark** on **Databricks Serverless** to predict student dropout risk from large-scale e-learning interaction data.

The project uses the **EdNet KT3** dataset containing **89.19 million interaction records** from **297,915 students** collected on an AI-based tutoring platform between August 2018 and November 2019. Because the module focuses on **Big Data**, the project emphasises:

- distributed data ingestion and cleaning with **Apache Spark**
- scalable feature engineering using **Spark DataFrames and Spark SQL**
- unsupervised clustering with **Spark MLlib K-Means**
- supervised classification with **scikit-learn** (hybrid architecture — see rationale below)
- advanced **SparkSQL** window functions throughout (RANK, NTILE, PARTITION BY, cumulative SUM OVER)
- end-to-end pipeline design with no data leakage

---

## Team Members

| # | Name | Student ID |
|---|------|------------|
| 1 | Joseph Rithik Gomes | 25065831 |
| 2 | Nabeel Mohammed | 25044419 |
| 3 | Nimisha Mariam Samuel | 25227866 |
| 4 | Sowmya Sree Vempalli Suresh | 25028405 |
| 5 | Nandhitha Reddy Anugu | 25299093 |
| 6 | Swathi Murali | 25243586 |

**Module:** CS6502 – Applied Big Data and Visualization  
**Institution:** University of Limerick  
**Group:** 7

---

## Problem Statement

Raw e-learning interaction data is large, fine-grained, and not directly suitable for dropout prediction. The challenge is to:

1. ingest and clean 89+ million interaction records in a distributed environment,
2. aggregate interactions into 10 student-level behavioural features using Spark,
3. construct a data-driven dropout label without leakage,
4. train and evaluate six classification models,
5. validate findings using unsupervised clustering,
6. present actionable insights for early intervention.

The target variable is **`dropout_risk`** — a binary label (0 = active, 1 = at-risk) derived from a recency-based threshold.

---

## Dataset Summary

The dataset is the **EdNet KT3** large-scale learning interaction dataset.

| Metric | Value |
|--------|-------|
| Raw interaction records | 89,270,654 |
| Records after cleaning | 89,194,876 |
| Records removed | 75,778 (0.085%) |
| Unique students | 297,915 |
| Unique learning items | 29,498 |
| Distinct content sources | 8 |
| Observation window | 2018-08-27 to 2019-11-27 (15 months) |
| Raw columns | 7 (timestamp, action_type, item_id, source, user_answer, platform, user_id) |
| Environment | Databricks Community Edition (Serverless compute) |

### Content Source Distribution

| Source | Interactions | Percentage |
|--------|-------------|------------|
| sprint | 64,616,840 | 72.4% |
| my_note | 7,033,452 | 7.9% |
| diagnosis | 6,621,642 | 7.4% |
| adaptive_offer | 5,122,323 | 5.7% |
| review_quiz | 3,620,216 | 4.1% |
| review | 803,141 | 0.9% |
| tutor | 722,038 | 0.8% |
| archive | 655,224 | 0.7% |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Distributed processing | Apache Spark (PySpark) |
| SQL analytics | Spark SQL |
| Clustering | Spark MLlib / scikit-learn KMeans |
| Supervised classification | scikit-learn |
| Feature engineering | Spark DataFrames |
| Visualisation | Matplotlib, Seaborn |
| Platform | Databricks Community Edition (Serverless) |
| Language | Python 3 |

---

## Architecture: Why the Hybrid Spark + scikit-learn Approach

Spark was used for all data processing, feature engineering, SQL analysis, and K-Means clustering. scikit-learn was used for supervised model training.

**Rationale:** After student-level aggregation, the feature matrix is 297,915 rows × 10 features — manageable in memory. On Databricks Serverless, every Spark `.fit()` call spawns a full DAG planning cycle with shuffle coordination and executor warmup. On a free-tier cluster this overhead costs more time than the actual computation (>30 minutes per model observed). scikit-learn eliminates this by training in-memory in seconds.

PySpark retained responsibility for:
- ingesting and cleaning 89M records
- all Spark SQL window-function analytics
- VectorAssembler → StandardScaler pipeline for K-Means input
- K-Means clustering (238,332-row training set)

scikit-learn handled:
- all six supervised classifiers
- StandardScaler for classification (fit on training rows only, no leakage)
- train/test split with stratification

---

## Project Workflow

### Task 1 — Data Ingestion and Cleaning (Cells 1–31)

- Loaded EdNet KT3 CSV into Spark (89,270,654 rows, 7 columns)
- Converted Unix millisecond timestamps to readable datetime
- Removed nulls on `user_id`, `timestamp`, `action_type` — 0 rows removed (no nulls in full dataset)
- Removed duplicates — 75,778 rows (0.085%) removed
- Standardised categorical columns with `lower()` + `trim()`
- Registered Spark SQL view `ednet` for analytical queries
- Identified 0 students with only a single interaction (all students have multiple records)

### Task 2 — Exploratory Data Analysis (Cells 32–49)

- Summary statistics and schema inspection
- Dropout rate by platform diversity (all students used a single platform)
- Activity tier breakdown using `CASE WHEN` (Low < 50, Medium 50–499, High 500+)
- Top 15 most engaged students using `RANK()` window function
- Temporal activity patterns (hourly and monthly)
- Correlation heatmap — 10 ML features + dropout_risk (N=297,915)

### Task 3 — Feature Analysis (Cells 50–71)

- Pearson correlation of each feature with dropout_risk
  - Strongest: `student_lifespan` (–0.1544)
  - Followed by: `active_days` (–0.1123), `learning_sessions` (–0.0864)
- IQR-based outlier detection on `total_interactions` — 51,063 outliers (17.1%)
- Activity segmentation using `NTILE()` window function
- Platform-group deviation using `AVG() OVER (PARTITION BY)`
- Spark Pipeline: VectorAssembler → SparkStandardScaler (fitted on train only)
  - Training split: 238,629 rows | Test split: 59,286 rows

### Task 4 — Machine Learning (Cells 72–95)

#### Dropout Label Construction
- Reference date: 2019-11-27 (most recent timestamp in dataset)
- Threshold: **314 days** (70th percentile of `days_since_last_activity`)
- Active students (label=0): **208,454 (69.97%)**
- At-risk students (label=1): **89,461 (30.03%)**
- Sanity check: active students average 3.3× more interactions and 2.3× more active days

#### Feature Engineering (10 features)

| Feature | Description |
|---------|-------------|
| total_interactions | Total number of interaction records per student |
| active_days | Number of distinct calendar days with any activity |
| learning_sessions | Count of 'enter' action type events |
| question_responses | Count of 'respond' action type events |
| submissions | Count of 'submit' action type events |
| unique_items | Number of distinct learning items accessed |
| unique_sources | Number of distinct content sources used |
| platforms_used | Number of distinct platforms used |
| avg_interactions_per_day | total_interactions ÷ active_days |
| student_lifespan | Days between first and last interaction |

#### Supervised Classification (scikit-learn, Cell 78)

Stratified 80/20 split: 238,332 training students | 59,583 test students  
`class_weight={0:1.0, 1:1.5}` applied to all models to address 70/30 imbalance.

| Model | F1-Score | Accuracy | AUC-ROC | Tuned Parameters |
|-------|----------|----------|---------|-----------------|
| ★ Random Forest | **0.6329** | 0.6935 | 0.6378 | n_estimators=200, max_depth=12, class_weight={0:1,1:1.5} |
| GBT | 0.5766 | 0.6996 | **0.6392** ★ | n_estimators=150, learning_rate=0.1, max_depth=5 |
| Decision Tree | 0.6283 | 0.6915 | 0.6215 | max_depth=6, class_weight={0:1,1:1.5} |
| MLP (Neural Net) | 0.5761 | 0.6997 | 0.6209 | hidden_layer_sizes=(32,16), alpha=0.01 |
| Logistic Regression | 0.5762 | 0.6995 | 0.6039 | C=1.0, class_weight={0:1,1:1.5} |
| SVM (LinearSVC) | 0.5761 | 0.6995 | 0.6035 | C=0.1, class_weight={0:1,1:1.5} |

★ = best in category. Evaluated on 59,583-student held-out test set.

**Key finding:** At baseline, all models predicted the majority class (F1≈0.5761). `class_weight` was the critical tuning driver — Random Forest showed the largest improvement (+0.0568 F1). Ensemble methods (RF, GBT, DT) outperformed all linear models on F1, confirming non-linear engagement patterns.

#### Feature Importance (GBT — Cell 75)

| Rank | Feature | GBT Importance |
|------|---------|---------------|
| 1 | student_lifespan | 56.1% |
| 2 | unique_items | 13.9% |
| 3 | unique_sources | 8.1% |
| 4–10 | Remaining features | ~22% combined |

`student_lifespan` dominates — total engagement duration is a stronger dropout signal than any single session or volume metric.

#### K-Means Clustering (Cells 86–91)

Implemented using `sklearn.cluster.KMeans` on `X_train_sc` (238,332 students). Silhouette scores computed on a 5,000-row subsample to avoid O(N²) cost on 238K rows.

| k | WCSS | WCSS Drop | Silhouette |
|---|------|-----------|-----------|
| 2 | 1,508,635.30 | — | **0.8648** ★ |
| 3 | 1,095,181.89 | 413,453.41 (largest) | 0.7695 |
| 4 | 914,431.21 | 180,750.68 | 0.7194 |
| 5 | 762,410.56 | 152,020.65 | 0.7291 |

k=2 selected — highest silhouette and largest WCSS elbow.

| Cluster | Students | Dropout Rate | Profile |
|---------|----------|-------------|---------|
| Cluster 0 — The Engine | 5,242 (2.2%) | 6.51% | Extreme power users (+4.77σ interactions) — highly retained |
| Cluster 1 — The Silent Majority | 233,090 (97.8%) | 30.56% | Sparse learners, near-zero activity — typical at-risk population |

The 4.7× difference in dropout rates across unsupervised groups independently validates the recency-based supervised label.

---

## Key Results Summary

| Finding | Value |
|---------|-------|
| Dataset size | 89,194,876 cleaned interactions, 297,915 students |
| Dropout threshold | 314 days (70th percentile of inactivity) |
| At-risk students | 89,461 (30.03%) |
| Best F1-score | Random Forest — 0.6329 |
| Best AUC-ROC | GBT — 0.6392 |
| Most important feature | student_lifespan (56.1% GBT importance) |
| K-Means silhouette | 0.8648 at k=2 |
| High-engagement cluster dropout rate | 6.51% (Cluster 0, 5,242 students) |
| General-learner cluster dropout rate | 30.56% (Cluster 1, 233,090 students) |

---

## Notebook Structure

The project is contained in a single notebook `CS6502_EdNet_Project_V2.ipynb` organised into four tasks:

| Task | Cells | Description |
|------|-------|-------------|
| Task 1 | 1–31 | Data ingestion, cleaning, and Spark SQL analysis |
| Task 2 | 32–49 | Exploratory data analysis and visualisations |
| Task 3 | 50–71 | Feature analysis, correlation, outlier detection, and scaling pipeline |
| Task 4 | 72–95 | ML models, K-Means clustering, model comparison, and feature importance |

---

## How to Run

### Prerequisites

- Databricks Community Edition (Serverless compute)
- EdNet KT3 dataset loaded into a Databricks Volume at `/Volumes/workspace/default/ednetdata/`
- Python packages: `pyspark`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

### Execution

Run all cells sequentially from Cell 1. The notebook is self-contained — all imports are consolidated in Cell 2 (global imports cell).

> **Note:** Cell 65 runs the Spark preprocessing pipeline (VectorAssembler + StandardScaler). If running from Task 4 only, re-run Cell 65 first to recreate `scaled_train_data` and `scaled_test_data`. Cells 77–78 (sklearn training) complete in under 30 seconds.

---

## Why This Is a Big Data Project

This project is not a standard single-machine ML task. It qualifies as a Big Data project because:

- the raw dataset contains **89+ million records** requiring distributed processing
- all ingestion, cleaning, and feature engineering uses **Spark distributed computation**
- **Spark SQL** with advanced window functions (RANK, NTILE, PARTITION BY, cumulative SUM OVER) processes aggregations across 89M rows
- **K-Means clustering** runs on the full 238K-row scaled training set via Spark MLlib / sklearn
- the pipeline is designed with **no data leakage** — scalers fitted on training data only
- the project demonstrates genuine **Big Data challenges**:
  - O(N²) silhouette score infeasible on 238K rows — resolved by subsampling
  - Spark job scheduling overhead making supervised training impractical — resolved by sklearn migration
  - 89M-row deduplication and null handling in distributed memory

---

## Outputs

The project produces:

- cleaned and feature-engineered student-level Spark DataFrames
- Spark SQL analytical query outputs and visualisations
- correlation heatmap across 10 ML features
- six trained classification models with baseline and tuned comparisons
- confusion matrices for all six models on a 59,583-student test set
- K-Means elbow and silhouette analysis (k=2 to 5)
- cluster profiles and dropout rate validation
- GBT feature importance and LR odds ratio analysis
- RF vs GBT feature importance comparison chart

---

## Future Improvements

- incorporate temporal engagement features (recency acceleration, session gap lengths)
- add external signals such as curriculum difficulty or assessment performance
- deploy as a real-time early-warning system with streaming Spark
- apply deep learning sequence models (LSTM) on raw interaction sequences
- build an interactive Databricks dashboard for instructor intervention support

---

## Module Context

**Module:** CS6502 – Applied Big Data and Visualization  
**University:** University of Limerick  
**Group:** 7

This project reflects module themes including distributed data processing, Spark-based analytics, machine learning at scale, big data visualisation, and end-to-end pipeline design.

---

## Acknowledgement

This project was developed for academic purposes as part of the CS6502 Applied Big Data and Visualization module. All data processing and modelling decisions were made with reference to published literature on educational dropout prediction and large-scale Spark deployment constraints.
