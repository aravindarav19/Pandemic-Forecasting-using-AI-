# Pandemic-Forecasting-using-AI-


---

#### Project Title:  
**Pandemic Forecasting Using AI: GCNN and LSTM for COVID-19 Prediction**

---

#### Project Description:  
This project focuses on forecasting the progression of the COVID-19 pandemic using advanced AI techniques. By leveraging Graph Convolutional Neural Networks (GCNN) and Long Short-Term Memory (LSTM) networks, the study models spatial and temporal dependencies to predict infection rates effectively.

---

#### Project Goals:  
1. Develop a GCNN model to capture spatial relationships (geographic spread).  
2. Implement an LSTM model to forecast temporal trends (time-series progression).  
3. Compare the efficacy of both models and determine their strengths in pandemic forecasting.  
4. Enhance public health decision-making by providing actionable insights based on predictive models.

---

#### Files in the Repository:  
1. **GCNN AND LSTM MODEL .ipynb**: Jupyter Notebook containing the complete codebase for the GCNN and LSTM models.
2. **Dissertation.pdf**: The complete dissertation providing the theoretical and experimental foundation of the project.
3. **README.md**: This file.

---

#### Prerequisites:  
1. **Python (v3.8 or higher)**  
2. Libraries:  
   - TensorFlow or PyTorch  
   - NumPy  
   - Pandas  
   - Matplotlib  
   - Scikit-learn  
   - PyTorch Geometric (for GCNN)

---

#### Data Overview:  
- The dataset includes COVID-19 case counts, death counts, and population statistics across U.S. counties.  
- Features: Case counts, death counts, population, geographical data (county and state), and temporal data (week index).  
- Data preprocessing steps:  
  - Aggregation of daily data into weekly totals.  
  - Handling missing values using interpolation or removal.  
  - Normalization of population data.  
  - Pivoting data into sequential format for LSTM.

---

#### Methodology:  
1. **GCNN Implementation**:  
   - Constructed a graph where counties are nodes, and edges represent inter-county relationships (e.g., shared state).  
   - Aggregated spatial information using graph convolutional layers.  
   - Predicted weekly COVID-19 cases based on node features and their spatial neighbors.

2. **LSTM Implementation**:  
   - Built sequential models to predict weekly COVID-19 case progression.  
   - Captured temporal trends using sliding windows of historical data.

3. **Evaluation Metrics**:  
   - Accuracy, RMSE (Root Mean Squared Error), and R-squared for regression tasks.  
   - Precision, Recall, and F1-score for binary classification tasks.

---

#### How to Run the Code:  
1. Clone the repository or download the `GCNN AND LSTM MODEL .ipynb`.  
2. Install required libraries via pip:  
   ```bash
   pip install numpy pandas matplotlib torch torch-geometric scikit-learn
   ```
3. Run the Jupyter Notebook (`GCNN AND LSTM MODEL .ipynb`) for step-by-step execution.  
   - Ensure you have a compatible GPU for faster GCNN training.  

---

#### Key Results:  
- **GCNN** outperformed **LSTM** in predicting COVID-19 spread, achieving:  
  - Accuracy: 93%  
  - RMSE: 0.96  
  - R-squared: 2.41  
- LSTM showed stronger precision in some cases but struggled to model geographic dependencies.

---

#### Future Work:  
1. Expand the GCNN model to include global datasets and cross-border data.  
2. Integrate real-time mobility and social behavior data for enhanced forecasting.  
3. Apply the methodology to other infectious diseases.

---

#### Acknowledgments:  
Thanks to the University of Liverpool, my supervisor Dr. Simona Capponi, and contributors for their guidance and support.
