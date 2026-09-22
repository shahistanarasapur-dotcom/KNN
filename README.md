# Iris Flower Classification using K-Nearest Neighbors (KNN)

This repository contains an end-to-end Machine Learning pipeline utilizing the **K-Nearest Neighbors (KNN)** algorithm to classify iris species based on physical measurements.

---

## 📌 Overview

The K-Nearest Neighbors (KNN) algorithm is an instance-based, non-parametric, lazy learning algorithm. Instead of constructing a generalized model during training, KNN memorizes the dataset and performs computation at query time by measuring distances between target points and stored instances.

### KNN Algorithm Workflow
1. **Memorization:** Store the complete training dataset[cite: 1].
2. **Distance Calculation:** Compute the distance metric (e.g., Euclidean distance) between the test point and all training instances[cite: 1].
3. **Neighbor Identification:** Select the $K$ closest neighbors based on the computed distances[cite: 1].
4. **Majority Voting:** Assign the class label based on majority vote among the $K$-nearest neighbors[cite: 1].

> **Best Practice:** Select an odd number for $K$ (e.g., $K=3$) to avoid tied votes in binary or multi-class classification scenarios[cite: 1].

---

## 📊 Dataset Specification

The project evaluates performance on the classic **Iris Dataset** loaded via `scikit-learn`[cite: 1].

* **Total Samples:** 150 instances[cite: 1]
* **Target Classes:** 3 (`setosa`, `versicolor`, `virginica`)[cite: 1]
* **Features:**
  * Sepal Length ($\text{cm}$)[cite: 1]
  * Sepal Width ($\text{cm}$)[cite: 1]
  * Petal Length ($\text{cm}$)[cite: 1]
  * Petal Width ($\text{cm}$)[cite: 1]

---

## 🚀 Quickstart & Pipeline Implementation

### Prerequisites
Ensure you have Python installed alongside the required dependencies:

```bash
pip install pandas matplotlib scikit-learn# KNN
