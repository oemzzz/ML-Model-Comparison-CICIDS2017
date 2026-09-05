# Comparative Evaluation of Ensemble and Tree-Based Machine Learning Algorithms for Network Intrusion Detection

[![DOI](https://img.shields.io/badge/DOI-10.30564%2Fjeis.v7i2.12299-blue)](https://doi.org/10.30564/jeis.v7i2.12299)

This repository contains the implementation and experimental setup for our research on network intrusion detection. The primary focus of this study is to overcome the severe class imbalance problem in network traffic data, specifically targeting the detection of minority attack classes such as Botnet, Infiltration, and Web Attacks.

## Project Overview
Modern Security Operations Centers (SOCs) require robust and scalable models for real-time threat detection. In this project, we applied a comprehensive data preprocessing architecture on the **CIC-IDS-2017** dataset and evaluated several machine learning algorithms to determine the most reliable model against asymmetric cyber threats.

## Models Evaluated
* **XGBoost** (Extreme Gradient Boosting)
* **LightGBM** (Light Gradient Boosting Machine)
* **Random Forest**
* **Decision Tree**

## Key Findings
* **XGBoost** outperformed all other evaluated tree-based models, achieving an **Accuracy of 99.89%** and a **Macro F1-score of 0.8903**.
* The experimental results prove XGBoost's high generalization capacity, demonstrating its effectiveness as a reliable machine learning model for real-time threat detection in SOC environments.

## Citation
If you find this repository or our research useful, please consider citing our published paper:

> **Atakan Özçelebi, Vedat Marttin.** "Comparative Evaluation of Ensemble and Tree-Based Machine Learning Algorithms for Network Intrusion Detection."  
> **DOI:** [https://doi.org/10.30564/jeis.v7i2.12299](https://doi.org/10.30564/jeis.v7i2.12299)
