# Chest X-Ray Medical Diagnosis with Deep Learning

A deep learning-based chest X-ray classifier built using DenseNet121 architecture for medical image diagnosis. This project includes both a Jupyter notebook for model training and a Flask web application for inference.

![Chest X-Ray Analysis](Grab-CAM.png)

## Overview

This project explores medical image diagnosis by building a state-of-the-art chest X-ray classifier using Keras and TensorFlow. The model can detect multiple pathologies from chest X-ray images using the NIH Chest X-ray dataset.

## Features

- **Transfer Learning**: Uses pre-trained DenseNet121 model
- **Multi-label Classification**: Detects multiple pathologies simultaneously
- **Class Imbalance Handling**: Implements techniques to handle imbalanced medical datasets
- **Performance Metrics**: Computes AUC-ROC curves for diagnostic performance
- **Visualization**: GradCAM visualization for model interpretability
- **Web Interface**: Flask-based web application for easy inference

## Project Structure

```
.
├── app.py                          # Flask web application
├── Chest_X-Ray_DenseNet121_Classifier_.ipynb  # Training notebook
├── util.py                         # Utility functions
├── public_tests.py                 # Test functions
├── requirements.txt                # Python dependencies
├── flask_requirements.txt          # Flask-specific dependencies
├── start_server.sh                 # Server startup script
├── data/
│   └── nih/                        # NIH dataset
│       ├── images-small/           # X-ray images
│       ├── train-small.csv         # Training labels
│       ├── valid-small.csv         # Validation labels
│       └── test.csv                # Test labels
├── models/
│   └── nih/                        # Saved models
├── images/                         # Project images
├── static/                         # Web app static files
│   ├── css/
│   └── js/
└── templates/                      # HTML templates
    └── index.html
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "Chest X-Ray Medical Diagnosis with DL (1)"
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Open and run the Jupyter notebook:
```bash
Jupyter Notebook Chest_X-Ray_DenseNet121_Classifier_.ipynb
```

The notebook covers:
- Data preprocessing and preparation
- Transfer learning with DenseNet121
- Handling class imbalance
- Model evaluation with AUC-ROC metrics
- GradCAM visualization

### Running the Web Application

1. Install Flask dependencies:
```bash
pip install -r flask_requirements.txt
```

2. Start the server:
```bash
bash start_server.sh
```

Or manually:
```bash
python app.py
```

3. Open your browser and navigate to `http://localhost:5000`

## Model Architecture

The model uses **DenseNet121** as the base architecture:
- Pre-trained on ImageNet
- Custom classification head for multi-label prediction
- Global Average Pooling layer
- Dense output layer with sigmoid activation

## Dataset

The project uses the **NIH Chest X-ray Dataset**:
- Contains over 100,000 chest X-ray images
- 14 different pathology labels
- Images are 1024x1024 pixels
- Includes patient metadata

## Performance Metrics

The model is evaluated using:
- **AUC-ROC**: Area Under the Receiver Operating Characteristic curve
- **Precision and Recall**: For each pathology class
- **Confusion Matrix**: For detailed performance analysis

## Visualization

The project includes **GradCAM** (Gradient-weighted Class Activation Mapping) visualization to:
- Highlight regions of interest in X-ray images
- Provide interpretability for model predictions
- Help clinicians understand model decisions

## Technologies Used

- **Python 3.11**
- **TensorFlow/Keras**: Deep learning framework
- **NumPy & Pandas**: Data manipulation
- **Matplotlib & Seaborn**: Visualization
- **Flask**: Web application framework
- **scikit-learn**: Machine learning utilities

## Requirements

See `requirements.txt` for the full list of dependencies:
- tensorflow>=2.0.0
- keras
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is for educational purposes.

## Acknowledgments

- NIH Clinical Center for the Chest X-ray dataset
- DenseNet architecture by Huang et al.
- GradCAM visualization technique

## Contact

For questions or feedback, please open an issue in the repository.
