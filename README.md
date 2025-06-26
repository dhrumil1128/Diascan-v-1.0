# **Diascan-v-1.0**

## **Project Overview**

**Diascan-v-1.0** is an innovative **web application** designed to help individuals assess their risk for **diabetes** based on key health parameters. The application enables users to easily detect if they have diabetes or are at risk of developing it by entering specific health-related factors into the system. 


### How It Works:

Users input their personal health details, such as glucose level, age, blood pressure, BMI, insulin levels, and other relevant health data into the application. The web application processes this data using a **Decision Tree Classifier**, a machine learning model, which analyzes the input and predicts whether the user has diabetes or not. The system also provides insights and recommendations based on the analysis.

The web application is designed to be intuitive, with a user-friendly interface that allows users to enter their health data quickly and easily. It also features data visualization to help users better understand their health metrics and predictions.

### **The Goal**:
The primary goal of **VitaPulseDX** is to empower users by providing a simple and effective way to detect diabetes early, without needing to visit a healthcare facility. Early detection of diabetes can help users take timely action to manage their health and prevent the onset of more serious complications. 

By using this application, users can:
- **Monitor** their health status regularly.
- **Detect early-stage diabetes** based on relevant health metrics.
- **Make informed health decisions** by understanding their risk factors.
- **Empower themselves** to take control of their health and wellbeing.

## Features:

- User-friendly interface for data input
- Predicts diabetes based on the following factors:
  - Glucose
  - Blood Pressure
  - Age
  - Gender
  - Insulin
  - BMI
  - Diabetes Pedigree
- Data visualization for better understanding of the input and predictions
- Machine learning concepts, including the confusion matrix for model evaluation
- Built using the Flask framework for seamless integration of frontend and backend

## Tech Stack:
- **Frontend:** HTML, CSS, JavaScript, Bootstrap
- **Backend:** Python, Flask
- **Machine Learning Model:** Decision Tree Classifier
- **Libraries:** pandas, scikit-learn, matplotlib, seaborn

## Clone this Project

To clone this project to your local machine, follow the steps below:

### **Steps to Clone:**

1. Copy the Repository URL:  
   You can find the repository URL at the top of the GitHub page (e.g., `https://github.com/your-username/Diascan-v-1.0`).

2. Clone the Repository:  
   Open your terminal (or Git Bash) and run the following command:
   ```bash
   git clone https://github.com/your-username/Diascan-v-1.0.git
   ```

3. Navigate to the Project Folder:  
   Once the cloning process is complete, navigate to the project directory:
   ```bash
   cd Diascan-v-1.0
   ```

4. Install Dependencies:  
   You may need to install the required dependencies. Run the following command:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the Application:  
   Once the dependencies are installed, you can run the Flask application:
   ```bash
   python app.py
   ```

   The web application should now be running locally at `http://127.0.0.1:5000/` in your web browser.

## Contributing

1. Fork the repository.
2. Create a new branch for your changes (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request to the main repository.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


