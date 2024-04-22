import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def randomForestClassifier(inp_features):
    # Load dataset
    diab_df = pd.read_csv("diabetes.csv")

    # Separate features and target
    X = diab_df.drop("Outcome", axis=1)
    y = diab_df["Outcome"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict on input features
    input_features = pd.DataFrame([inp_features], columns=X.columns)
    result = model.predict(input_features)[0]
    return result

def gradientBoostingClassifier(inp_features):
    # Load dataset
    diab_df = pd.read_csv("diabetes.csv")

    # Separate features and target
    X = diab_df.drop("Outcome", axis=1)
    y = diab_df["Outcome"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Gradient Boosting (XGBoost) model
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict on input features
    input_features = pd.DataFrame([inp_features], columns=X.columns)
    result = model.predict(input_features)[0]
    return result


def supportVectorMachine(inp_features):
    # Load dataset
    diab_df = pd.read_csv("diabetes.csv")

    # Separate features and target
    X = diab_df.drop("Outcome", axis=1)
    y = diab_df["Outcome"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM model
    model = SVC(kernel='linear')
    model.fit(X_train_scaled, y_train)

    # Predict on input features
    input_features = np.array(inp_features).reshape(1, -1)
    input_scaled = scaler.transform(input_features)
    result = model.predict(input_scaled)[0]
    return result

def logisticRegression(inp_features):
    # Load dataset
    diab_df = pd.read_csv("diabetes.csv")

    # Separate features and target
    X = diab_df.drop("Outcome", axis=1)
    y = diab_df["Outcome"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Predict on input features
    input_features = np.array(inp_features).reshape(1, -1)
    result = model.predict(input_features)[0]
    return result

def naiveBayes(inp_features):

    diab_df = pd.read_csv("diabetes.csv")
    true_filter = (diab_df["Outcome"] == 1)
    diab_df.drop("Outcome",axis=1,inplace=True)

    true_array = diab_df[true_filter].to_numpy()
    false_array =  diab_df[~ true_filter].to_numpy()

    true_prior= len(true_array)/len(diab_df)
    means = true_array.mean(axis=0,keepdims=True)
    variance = true_array.var(axis=0,keepdims=True)


    means = np.concatenate((false_array.mean(axis=0,keepdims=True),means),axis=0)
    variance = np.concatenate((false_array.var(axis=0,keepdims=True),variance),axis=0)


    x=np.array(inp_features).reshape(1,1,8)
    p_val = np.exp(-0.5*(x-means)**2/variance)/np.sqrt(2*np.pi*(variance+1e-9))
    probs = (np.prod(p_val,axis=2))* np.array([1-true_prior,true_prior])[None,:]
    result = probs.argmax(axis=1)
    return result[0]


def ensemble_prediction(inp_features):
    # Call each classifier function and store the results in a list
    results = []
    results.append(randomForestClassifier(inp_features))
    results.append(gradientBoostingClassifier(inp_features))
    results.append(supportVectorMachine(inp_features))
    results.append(naiveBayes(inp_features))
    results.append(logisticRegression(inp_features)) 

    # Calculate the average of the results
    avg_result = np.mean(results)

    # Round the average result to the nearest integer (0 or 1)
    final_result = round(avg_result)

    return final_result


# Example usage
# inp_features = [1, 148, 72, 35, 0, 33.6, 0.627, 50]  # Example input features

# result_random_forest = randomForestClassifier(inp_features)
# print("Random Forest Classifier Result:", result_random_forest)

# result_gradient_boosting = gradientBoostingClassifier(inp_features)
# print("Gradient Boosting (XGBoost) Result:", result_gradient_boosting)


# result_svm = supportVectorMachine(inp_features)
# print("SVM Result:", result_svm)

# result_svm = naiveBayes(inp_features)
# print("naiveBayes Result:", result_svm)

