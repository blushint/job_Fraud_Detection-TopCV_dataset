# TopCV Fraud Detection

## Overview

This project develops a machine learning pipeline for detecting fraudulent job postings on the TopCV platform. By analyzing job listing content (e.g., titles, descriptions, requirements), the system identifies postings that exhibit suspicious patterns indicative of scams.

The solution covers the full data science workflow, from data acquisition and preprocessing to feature engineering, model training, and evaluation.

---
<img width="1442" height="907" alt="image" src="https://github.com/user-attachments/assets/13c92763-5006-482c-99c6-169082a9a07f" />
The flowchart illustrates the end-to-end pipeline: data collection from TopCV using Scrapy and Selenium, storage in Hadoop/HDFS, preprocessing with Spark, feature extraction via PhoBERT, oversampling with SMOTE, model training (KNN, Decision Tree, SVM), and evaluation with metrics like accuracy, precision, recall, F1-score, and AUC-ROC, augmented by rule-based fraud scoring.

## Objectives

- Detect fraudulent users based on behavioral data
- Build interpretable and high-performance classification models
- Evaluate the model using precision, recall, F1-score, and ROC-AUC
- Recommend potential improvements and next steps for deployment

---

## Dataset

The dataset used in this project was collected by crawling publicly available job postings on the TopCV platform. Due to terms of service and data privacy considerations, the dataset is not shared in this repository. No sample data is provided either.

However, a crawling script is included in the project to help you reproduce the dataset generation process independently, subject to compliance with the data source’s terms of use.

## Repository Structure

- `G5_TopCV_Fraud_Detection.ipynb`: Jupyter notebook containing the full implementation pipeline
- `top-cv-job-crawler`: Sscript used to collect behavioral data from public TopCV pages
- `requirements.txt`: List of required Python packages
- `README.md`: Project documentation

## How to Run

1. Clone the repository:
   git clone (https://github.com/blushint/job_Fraud_Detection-TopCV_dataset.git)

2. Install dependencies:
   pip install -r requirements.txt

3. (Optional) Generate your dataset by running:
   top-cv-job-crawler

4. Launch the notebook and follow the workflow:
   jupyter notebook G5_TopCV_Fraud_Detection.ipynb

## Technologies Used

- Python 3.10+
- pandas, numpy, matplotlib, seaborn
- Transformers (PhoBERT), PyTorch, imbalanced-learn (SMOTE)
- scikit-learn, Matplotlib
- Jupyter Notebook

## Results

The final models (KNN, Decision Tree, SVM) achieved effective fraud detection after SMOTE oversampling and PhoBERT embeddings. 
Key indicators of fraud included anomalous textual patterns in descriptions and requirements. Models were evaluated with detailed metrics; visualizations in notebook.

## Limitations and Future Work

- Dataset may contain labeling noise; further refinement of fraud rules recommended
- Model generalization should be tested on evolving platform data
- Future development may include real-time API integration, ensemble methods, or advanced NLP techniques

## Acknowledgments

We would like to express our deepest gratitude to **Mr. Nguyễn Mạnh Tuấn**, who not only guided us throughout this project but also inspired us to approach each problem with critical thinking and scientific rigor.
His support, encouragement, and valuable feedback were instrumental in shaping our understanding of real-world machine learning applications.  


## License

This repository is for educational and non-commercial research purposes only. The authors do not publish, distribute, or license any dataset. Use of the crawling script must comply with the source website’s terms and applicable laws.

## Authors

Developed by Group 5 – BigData Project

