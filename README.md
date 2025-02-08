# Wireless Network Time Series & PDR Prediction for building NDTs

This project focuses on collecting time series data from a wireless network, analyzing link quality, and training models to predict the Packet Delivery Ratio (PDR) using different approaches.  

## Code Structure  

### `code/expe/` - Experiment Firmware & Execution Scripts  

- **`broadcast-example.c`**: Firmware for M3 nodes in FIT IoT-Lab, handling broadcast communication.  
- **`exp.sh`**: Creates and configures an experiment, assigning firmware and parameters.  
- **`run-exp.sh`**: Calls `exp.sh` with experiment parameters such as packet size.  

### `code/` - Data Processing, Modeling & Visualization  

- **`generate_series.py`**: Converts experiment logs into structured JSON time series.  
- **`plot-links-comprehensive.py`**: Visualizes time series data for all links.  
- **`regression-best-model-continuous.py`**:  
  - Trains various models on the generated series.  
  - Applies a calibration process using a **single** selected model for testing.  
- **`regression-adaptive-model-continuous.py`**:  
  - Trains multiple models and dynamically switches between them for prediction.  
  - Applies an **adaptive calibration** strategy during testing.  
- **`plot-metrics-general-continuous.py`**: Plots MAE results for both approaches across all links as the prediction step increases.  
- **`plot-metrics-per-cluster-continuous.py`**:  
  - Groups links into clusters based on quality (mean PDR).  
  - Plots MAE results for both approaches, tracking performance across increasing prediction steps.  

### `data/` - Experiment Logs  

- **`logs-iotj-24h.txt`**: Raw logs from a **24-hour FIT IoT-Lab experiment**, used for generating time series.  

## Usage  

### 1. Run the Experiment  
   - Configure and launch an experiment with `run-exp.sh`, specifying parameters.  
   - Firmware executes on M3 nodes, collecting wireless communication data.  

### 2. Process Logs & Generate Time Series  
   - Convert logs into structured JSON using `generate_series.py`.  

### 3. Train & Evaluate Models  
   - Train models using `regression-best-model-continuous.py` or `regression-adaptive-model-continuous.py`.  
   - Compare approaches with `plot-metrics-general-continuous.py` and `plot-metrics-per-cluster-continuous.py`.  

### 4. Visualize Results  
   - Use `plot-links-comprehensive.py` to analyze link quality trends.  

## Dependencies  

- **Contiki-NG** for network experimentation.  
- **FIT IoT-Lab** for deployment.  
- **Python (NumPy, Pandas, Matplotlib, Scikit-learn)** for data processing & modeling.  