# Intrusion Detection System (IDS) Using Machine Learning and Deep Learning

## Overview
This project aims to develop an advanced **Intrusion Detection System (IDS)** using **Machine Learning (ML)** and **Deep Learning (DL)** techniques to detect and classify network attacks. The IDS is trained and evaluated using two prominent datasets, **NSBW-NB15** and **CICIDS 2018**, which provide comprehensive labeled data representing both normal and malicious network traffic. By implementing various algorithms, this IDS can effectively detect and mitigate modern cyber threats, enhancing network security.

## Key Features
- **Dual Dataset Integration:** Utilizes both the NSBW-NB15 and CICIDS 2018 datasets to improve model robustness and generalizability.
- **Comprehensive Attack Detection:** The IDS can detect a wide range of attack categories, including DoS (Denial of Service), DDoS (Distributed Denial of Service), Web-based attacks, Brute Force, Botnets, and more.
- **Multiple ML and DL Models:** Includes traditional machine learning models (Random Forest, Decision Tree, SVM) and deep learning architectures for advanced detection.
- **Thorough Model Evaluation:** Uses detailed metrics like Precision, Recall, F1-Score, and ROC AUC to assess model performance.

## Datasets

### 1. **NSBW-NB15 Dataset**
The **NSBW-NB15** dataset is a modern replacement for older datasets like KDDCup99, addressing issues of redundancy and imbalance. It simulates normal and abnormal network traffic, containing various attack types, including Exploits, Fuzzers, Backdoors, and DoS attacks. It is composed of over 2.5 million records of network flow data.

- **Features:**
  - 49 features for network flow analysis.
  - Types of attacks: DoS, Fuzzers, Generic, Reconnaissance, Shellcode, and Worms.

### 2. **CICIDS 2018 Dataset**
The **CICIDS 2018** dataset is designed to simulate modern attack behaviors. It represents different real-world attack scenarios, such as DDoS, Brute Force, Infiltration, Botnet, and more. This dataset contains comprehensive network traffic data from simulated environments and is highly suited for contemporary intrusion detection research.

- **Features:**
  - Includes over 80 features related to network traffic.
  - Attack types: Brute Force, DDoS, Botnet, Port Scanning, SQL Injection, etc.

## Models Implemented

### Machine Learning Models:
- **Random Forest:** A powerful ensemble method that builds multiple decision trees and merges them to obtain more accurate and stable predictions.
- **Decision Trees:** Simple and interpretable models that work well for classifying network traffic based on feature importance.
- **Support Vector Machine (SVM):** A model that is effective in high-dimensional spaces for classifying both normal and malicious traffic.

### Deep Learning Models:
- **Deep Neural Networks (DNN):** Implemented to learn complex patterns in network traffic data, enhancing the detection of sophisticated attacks.
- **LSTM (Long Short-Term Memory):** Suitable for sequential data processing, LSTM models were used to capture temporal patterns in network traffic to improve anomaly detection.

## Technology Stack
- **Programming Languages:** Python
- **Libraries:** 
  - Data Handling: Pandas, NumPy
  - ML/DL Frameworks: Scikit-learn, TensorFlow, Keras
  - Visualization: Matplotlib, Seaborn
- **Tools:** Jupyter Notebook, VS Code

## Model Evaluation
The models were evaluated using the following metrics:
- **Accuracy:** Overall correctness of the model.
- **Precision:** How many of the detected anomalies are actually malicious.
- **Recall:** How many actual malicious instances are detected.
- **F1-Score:** Harmonic mean of Precision and Recall, balancing the trade-off between the two.
- **ROC-AUC Score:** Measures the trade-off between true positive and false positive rates.


## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/username/IDS-Project.git
   cd IDS-Project
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebooks for training and evaluating models.

## Usage

1. **Data Preprocessing:** Clean, normalize, and transform the datasets.
2. **Model Training:** Train various machine learning and deep learning models on the preprocessed data.
3. **Model Evaluation:** Evaluate model performance using the test datasets and metrics.
4. **Prediction:** Use the trained models to classify network traffic as normal or malicious.

## Results and Discussion
Detailed analysis of the results, along with comparison between the performance of various models on different datasets, is included in the Jupyter Notebooks. The models demonstrated strong performance in detecting complex attack patterns from both NSL-NB15 and CICIDS 2018 datasets, providing a reliable intrusion detection system for modern cybersecurity challenges.

## Conclusion
This project demonstrates the effectiveness of using machine learning and deep learning approaches to build a robust intrusion detection system. The integration of two comprehensive datasets, NSL-NB15 and CICIDS 2018, ensures that the IDS can handle a wide range of modern network attacks, making it a valuable tool in enhancing network security.

## Future Work
- **Model Optimization:** Further optimize deep learning models for better accuracy and faster performance.
- **Real-Time Detection:** Implement real-time IDS by integrating this system into live network environments.
- **Additional Attack Scenarios:** Explore additional datasets and attack scenarios to further generalize the model.

## Contributing
Contributions to the project are welcome. Please feel free to submit a pull request or raise issues for discussion.

## License
This project is licensed under the **MIT License**.

## Contact
For any queries or feedback, feel free to contact me:
- **Email:** [surjanmukherjeeimp@gmail.com](mailto:surjanmukherjeeimp@gmail.com)
- **LinkedIn:** [Surjan Mukherjee](https://www.linkedin.com/in/surjan-mukherjee-90aa721bb/)

---
