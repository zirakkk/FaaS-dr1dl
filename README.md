# Serverless R1DL 🧠☁️

[![Python](https://img.shields.io/badge/Python-3.7-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.18.5-green.svg)](https://numpy.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Latest-orange.svg)](https://www.tensorflow.org/)
[![Google Cloud](https://img.shields.io/badge/GCP-Functions-blue.svg)](https://cloud.google.com/functions)

This repository implements the Rank-1 Dictionary Learning (R1DL) algorithm and Nilearn neuroimaging analysis as serverless cloud functions.

## 📋 Overview

The original R1DL algorithm was introduced in the paper [Scalable Fast Rank-1 Dictionary Learning for fMRI Big Data Analysis](http://www.kdd.org/kdd2016/subtopic/view/scalable-fast-rank-1-dictionary-learning-for-fmri-big-data-analysis) (KDD 2016). This project extends that work by implementing R1DL in a serverless architecture, enabling on-demand computation without maintaining dedicated servers.

## ⚙️ Serverless Implementations

### R1DL Cloud Function

The [`serverless/R1DL/main.py`](dr1dl-pyspark-master/dr1dl-pyspark-master/serverless/R1DL/main.py) provides a serverless implementation that:

- Loads data from Google Cloud Storage bucket (`tf-functions/R1DL/Original400k.txt`)
- Processes it using the R1DL algorithm with configurable parameters:
  - Sparsity: 0.2 (20% non-zero elements)
  - Dictionary atoms: 5
  - Convergence epsilon: 0.01
- Measures and reports execution time

Key function:
```python
def TheFunction(request):
    # Load data from cloud storage
    storage_client = storage.Client()
    bucket = storage_client.get_bucket("tf-functions")
    blob = bucket.blob("R1DL/Original400k.txt")
    downloaded_blob = blob.download_as_string()
    
    # Process with R1DL algorithm
    S = np.loadtxt(StringIO(downloaded_blob.decode("utf-8")))
    D, Z = r1dl(S, 0.2, 5, 0.01)
    
    # Report runtime
    runtime = stop - start
    print("Code run time: ", runtime)
    
    return "All good"
```

### Nilearn Cloud Function

The [`serverless/Nilearn/main.py`](dr1dl-pyspark-master/dr1dl-pyspark-master/serverless/Nilearn/main.py) demonstrates neuroimaging analysis in a serverless environment:

- Fetches ADHD dataset (6 subjects by default)
- Performs Independent Component Analysis with 20 components
- Extracts time series data with specific parameters:
  - Smoothing: 6mm FWHM
  - Temporal filtering: 0.01-0.1 Hz
  - TR: 2.5s
- Builds connectivity matrices and trains a neural network classifier
- Reports accuracy metrics and execution time

Key function:
```python
def nilearn(request):
    # Fetch ADHD dataset
    adhd_data = datasets.fetch_adhd(n_subjects=6)
    
    # Perform ICA and extract components
    canica = decomposition.CanICA(n_components=20)
    canica.fit(adhd_data["func"])
    
    # Build connectivity matrices
    correlation_matrices = correlation_measure.fit_transform(subjects)
    
    # Train neural network classifier
    classifier = keras.models.Sequential()
    classifier.add(keras.layers.Dense(16, activation="relu"))
    classifier.add(keras.layers.Dense(16, activation="relu"))
    classifier.add(keras.layers.Dense(1, activation="sigmoid"))
    
    # Report runtime
    runtime = stop - start
    print("Code run time: ", runtime)
    
    return "All good"
```

## 🚀 Getting Started

### Prerequisites

- Google Cloud Platform account
- Cloud Functions enabled
- Storage bucket for data
- Python 3.7+

### Dependencies

Each serverless function has its own requirements file:

**R1DL Requirements** ([`serverless/R1DL/requirements.txt`](dr1dl-pyspark-master/dr1dl-pyspark-master/serverless/R1DL/requirements.txt)):
```
nilearn
google-cloud-storage==1.16.1
sklearn
tensorflow
numpy==1.18.5
```

**Nilearn Requirements** ([`serverless/Nilearn/requirements.txt`](dr1dl-pyspark-master/dr1dl-pyspark-master/serverless/Nilearn/requirements.txt)):
```
nilearn
google-cloud-storage==1.16.1
sklearn
tensorflow
numpy==1.18.5
```

### Deployment

1. Create a Google Cloud Storage bucket
2. Upload your input data to the bucket (for R1DL function)
3. Deploy the cloud functions:

## 📊 Performance

The serverless implementation allows for on-demand computation without maintaining dedicated infrastructure.


## 📄 License

This project is licensed under the Apache License 2.0 - see the original repository's LICENSE file for details.


## 🙏 Acknowledgments

This work builds upon the original R1DL implementation by Milad Makkie, Xiang Li, Mojtaba Fazli, Tianming Liu, and Shannon Quinn as referenced in the original [publication](http://www.kdd.org/kdd2016/subtopic/view/scalable-fast-rank-1-dictionary-learning-for-fmri-big-data-analysis).
