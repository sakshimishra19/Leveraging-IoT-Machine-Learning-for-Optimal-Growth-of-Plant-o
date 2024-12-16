# **Leveraging IoT and Machine Learning for Plant Disease Prediction**

The project aims to predict plant diseases early in green gram crops using IoT sensors and machine learning. Environmental parameters like **temperature, humidity, rainfall, and soil moisture** are analyzed using a **Multi-Layer Perceptron (MLP)** model. The model is integrated with a **Streamlit-based interface** for real-time disease predictions using sensor data.

---

## **Features**
- **IoT-based Data Collection**: Real-time environmental data collection using sensors.  
- **ML Model**: Multi-Layer Perceptron achieving high accuracy.  
- **User Interface**: Streamlit app for user-friendly disease predictions.  
- **Environmental Parameters**: Temperature, humidity, rainfall, and soil moisture.  

---

## **Technologies Used**
- **Programming Language**: Python  
- **Libraries/Frameworks**:  
   - Machine Learning: Scikit-learn  
   - Interface: Streamlit  
   - Data Visualization: Matplotlib, Seaborn  
- **IoT Sensors**: For real-time data collection.  

---

## **Project Workflow**
1. **Data Collection**: Environmental data using IoT sensors and Kaggle dataset.  
2. **Exploratory Data Analysis (EDA)**: Feature relationships using pair plots and heatmaps.  
3. **Model Training**: MLP classifier trained for disease detection.  
4. **Model Integration**: Real-time sensor values are fed into the trained model.  
5. **User Interface**: Streamlit app provides real-time disease predictions.  

---

## **Screenshots**

### **Streamlit Interface**
![image](https://github.com/user-attachments/assets/da04810f-a3f7-4a5c-9275-931e58a5bb00)


### **Model Accuracy Results**
![image](https://github.com/user-attachments/assets/69e36efe-4bdc-4ed9-958b-04fe50e32b19)


---

## **Demo Video**
[Watch the Demo Video](./Project_K%20(1).mp4)


Click the image above to watch the full demo video.

---

## **How to Run the Project**
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/repo-name.git
   cd repo-name
   ```
2. Install dependencies:  
   ```bash
   pip install streamlit scikit-learn pandas matplotlib seaborn
   ```
3. Run the Streamlit app:  
   ```bash
   streamlit run app.py
   ```

---

## **Future Scope**
- Adding more parameters like **NPK levels** for improved accuracy.  
- Adapting the system for **other crops** with minimal changes.  
- Enhancing real-time monitoring using advanced IoT technologies.

---

## **Contact**
For queries or collaboration, reach out:  
- **Email**: sakshi.mishra@example.com  

