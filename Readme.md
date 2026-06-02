# AI Marketing Intelligence Streamlit App

This project is an AI-powered marketing intelligence dashboard built with Streamlit.  
It uses machine learning models (XGBoost, KMeans clustering, and a Scaler) to analyze customer data, segment users, and generate actionable marketing insights.  
The app includes interactive visualizations built with **Seaborn** and **Plotly**.

---
## Live Demo
https://ai-powered-marketing-tool-ejlnprlmbymhopx7q4kshk.streamlit.app/

##  Features

- Customer segmentation using **KMeans clustering**
- Predictive analytics using **XGBoost**
- Automated feature scaling using **StandardScaler**
- Interactive charts using **Plotly**
- Statistical visualizations using **Seaborn**
- CSV data upload and processing
- Clean, modern interface for marketing teams

---

##  Machine Learning Models Used

The app loads the following models:

- `scalar.pkl` – StandardScaler for preprocessing  
- `kmeansmodel.pkl` – KMeans clustering model  
- `xgboost.pkl` – XGBoost predictive model  
- `clusterlabels.pkl` – Cluster label mappings  
- `xgb features.pkl` – Feature list for XGBoost  

> ⚠️ **Note:**  
> These model files are stored externally (HuggingFace Hub) and downloaded at runtime.

---

## 📦 Installation

Clone the repository:https://github.com/Sh0917815/AI-POWERED-MARKETING-TOOL

git clone 
cd AI-POWERED-MARKETING-TOOL

Install dependencies:

pip install -r requirements.txt


---

## ▶️ Running the App

streamlit run app.py


---

## 🗂 Project Structure

project-name/
  app.py
  requirements.txt
  README.md
  data.csv
  .gitignore

Model files are stored externally and loaded dynamically.

---

## ☁️ Deployment

This app can be deployed on:

- Streamlit Cloud  
- HuggingFace Spaces  
- Render  
- AWS EC2  

---

## 🤝 Contributing

Pull requests are welcome.  
For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

MIT License
