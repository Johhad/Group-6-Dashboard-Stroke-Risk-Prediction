# NeuroPredict - Stroke Risk Prediction Dashboard

A comprehensive Streamlit dashboard for predicting stroke risk using machine learning algorithms and interactive data visualization.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dashboard Pages](#dashboard-pages)
- [Technologies Used](#technologies-used)
- [Contributors](#contributors)
- [License](#license)

## Features
- **Real-time Risk Prediction**: Input patient data and get instant stroke risk assessment
- **Interactive Visualizations**: Explore data through dynamic charts and graphs
- **What-If Analysis**: Test how lifestyle changes affect stroke risk
- **Comprehensive Analytics**: View descriptive and diagnostic statistics
- **User-Friendly Interface**: Clean, intuitive design for healthcare professionals

##Dashboard Pages
### 1. Home
- Overview of the dashboard
- Navigation guide
- Quick statistics
### 2. Descriptive Analysis
- Age distribution
- Gender distribution
- Risk factor prevalence
- Correlation analysis
### 3. Diagnostic Analysis
Statistical metrics
Risk factor analysis
Stroke rate analysis
Multi-variable relationships
### 4. Risk Prediction
- Patient data input form
- Real-time risk calculation
- Risk level visualization (Low/Medium/High)
- Risk gauge display
### 5. Preventive
- Interactive risk calculator
- Scenario simulation
- Prevention guidelines
- Lifestyle recommendations
6. About
- Project information
- Team details
- Model specifications
- Future enhancements
- Technologies Used
- Frontend Framework: Streamlit
- Programming Language: Python 3.8+
- Machine Learning: Scikit-learn
- Data Manipulation: Pandas, NumPy
- Visualization: Plotly
- Deployment: Streamlit Cloud

# 📁 Project Structure

...

stroke-risk-dashboard/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Git ignore file
│
├── data/                 # Data directory
│   ├── raw/             # Raw data files
│   └── processed/       # Processed data files
│
├── models/              # Trained models
│   └── stroke_model.pkl # Saved model file
│
├── utils/               # Utility functions
│   ├── data_processing.py
│   └── model_utils.py
│
└── assets/              # Static assets
    └── images/          # Images and icons

## 🔧 Configuration
Environment Variables
Create a .env file in the root directory:

```
STREAMLIT_THEME_PRIMARY_COLOR="#1f77b4"
STREAMLIT_THEME_BACKGROUND_COLOR="#ffffff"
```


### Streamlit Configuration
Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

## 📈 Model Information

### Features Used
1. Age
2. Gender
3. Hypertension
4. Heart Disease
5. Marriage Status
6. Work Type
7. Residence Type
8. Average Glucose Level
9. BMI
10. Smoking Status

### Algorithm
- **Model**: Random Forest Classifier
- **Accuracy**: ~92%
- **Validation**: 5-fold cross-validation

## 🚀 Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to share.streamlit.io
3. Connect your GitHub repository
4. Select the branch and main file (app.py)
5. Click Deploy

### Local Development

For development mode with auto-reload:
```bash
streamlit run app.py --server.runOnSave true
```

## 📝 Usage Guide

### Basic Usage
1. Navigate to the Risk Prediction page
2. Enter patient information
3. Click "Predict Risk"
4. View the risk assessment

### Data Input
- Ensure all fields are filled correctly
- Use appropriate ranges for numerical inputs
- Select from dropdown options for categorical data

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## 👥 Contributors

- Group 6 Members: Pamela, Pardon, Johannes, Chantale, Htet
- Data Science Team
- Healthcare Advisors
- UI/UX Designers

## 📄 License
This project is licensed under the Stockholm University License 

## ⚠️ Disclaimer
This tool is for educational and screening purposes only. It should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

## 📞 Contact
- Email: info@neuropredict.com
- GitHub: Group-6-Dashboard: [Group-6-Dashboard](https://github.com/pardonm6/Group-6-Dashboard-Stroke-Risk-Prediction)
- Issues: Report Issues: [Report issues](https://github.com/pardonm6/Group-6-Dashboard-Stroke-Risk-Prediction)

## 🙏 Acknowledgments

- Healthcare professionals who provided domain expertise
- Open-source community for tools and libraries
- Dataset providers for stroke risk data
- Streamlit team for the amazing framework

---
**Last Updated**: 2024 
**Version**: 1.0.0

1. Create the environment with python -m venv env
2. Activate the virtual environment for Python
  - If using Mac or Linux, type the command: source env/bin/activate
  - If using Windows:
  - First, set the Default Terminal Profile to CMD Terminal
  - Then, type in the CMD terminal: .\env\Scripts\activate.bat
3. Make sure that your terminal is in the environment (env) not in the global Python installation
4. Install required packages pip install -r ./requirements.txt
5. Check that everything is ok running streamlit hello
6. Stop the terminal by pressing Ctrl+C

### Execution
To run the dashboard execute the following command:

> streamlit run Dashboard.py
# If the command above fails, use:
> python -m streamlit run Dashboard.py
```


### Creating pre-trained models for the web dashboadr
⚠️ **NOTE**: In the predictive analytics tab, the web dashboard is looking for a pre-trained model in the folder assets/. The first time that you execute the application, it will show an error saying that such file does not exist. Therefore, you need to execute the notebook inside the folder jupyter-notebook/ to create the pre-trained model.
python -m streamlit run Dashboard.py
This logic resembles the expected pipeline, where the jupyter notebooks are used to iterate the data modeling part until a satisfactory trained model is created, and the streamlit scripts are only in charge of rendering the user-facing interface to generate the prediction for new data. In practice, the data science pipeline is completely independent from the web dashboard, and both are connected via the pre-trained model.

## Contributors
- Johannes Haddad: johanneshad_95@hotmail.com
- Pamela Castillo: pamela.abigail.castillo.gonzalez@ki.stud.se 
- Htet Wai Aung: htau6812@student.su.se 
- Pardon Runesu (pardonm6@gmail.com) 
- Chantale NzeggeMvele (nzeggechantal@gmail.com)











