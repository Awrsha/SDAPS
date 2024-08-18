# Intelligent Skin Disease Prediction System WebApp

## Overview
The Intelligent Skin Disease Prediction System WebApp is designed to assist users in identifying potential skin conditions through image analysis. This web application leverages machine learning models to provide diagnostic insights, helping users understand and address their skin health.

## Features
- **Image Upload**: Users can upload images of their skin conditions for analysis.
- **AI Diagnosis**: The application utilizes a trained deep learning model to predict potential skin conditions and their likelihood.
- **Detailed Results**: Displays the diagnosis, probability of malignancy, and provides both general and personalized advice based on the analysis.
- **Interactive UI**: Engaging user interface with animations, progress bars, and interactive elements.
- **Countdown Timer**: A countdown feature to encourage timely medical consultation.

## Directory Structure
```
- static/
  - models/
    - model_version_1.keras
  - uploads/
  - css/
    - style.css
  - js/
    - script.js
- templates/
  - index.html
  - result.html
- app.py
- docker-compose.yaml
- Dockerfile
- requirements.txt
```

## Getting Started

### Prerequisites
- Python 3.8+
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Awrsha/Intelligent-Skin-Disease-Prediction-System.git
   cd Intelligent-Skin-Disease-Prediction-System
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

### Docker Setup (Optional)
1. **Build the Docker image**
   ```bash
   docker build -t Intelligent-Skin-Disease-Prediction-System:latest .
   ```

2. **Run the Docker container**
   ```bash
   docker-compose up -d
   ```

## Usage
1. Open a web browser and navigate to `http://127.0.0.1:5000`.
2. Upload an image of the skin condition on the home page.
3. Wait for the analysis and view the detailed results on the result page.

## File Descriptions

### `app.py`
This is the main Flask application file that contains the routes and logic for handling requests and processing images.

### `static/models/model_version_1.keras`
This is the pre-trained deep learning model used for skin disease prediction.

### `static/uploads/`
This directory stores the uploaded images.

### `static/css/style.css`
Contains the custom CSS styles for the web application.

### `static/js/script.js`
Includes custom JavaScript for enhancing the interactivity of the web application.

### `templates/index.html`
The home page of the web application where users can upload images.

### `templates/result.html`
The result page where the diagnosis and recommendations are displayed.

### `docker-compose.yaml`
Configuration file for Docker Compose to set up the application in a containerized environment.

### `Dockerfile`
Defines the Docker image for the application.

### `requirements.txt`
Lists the Python dependencies required for the application.

## Contributing
We welcome contributions to improve the project. Please fork the repository and create a pull request with your changes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
We thank all contributors and the open-source community for their valuable work and resources that helped in developing this application.
