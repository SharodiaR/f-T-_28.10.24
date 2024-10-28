import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import pickle

# Load the Excel file
file_path = r"C:\\Users\\swast\\OneDrive\\Desktop\\Uncertainty Data.xlsx"
excel_data = pd.ExcelFile(file_path)

# Display the number of sheets
sheet_names = excel_data.sheet_names
print(f"Number of sheets: {len(sheet_names)}")

# Target temperatures for uncertainty factor calculation
target_temperatures = np.arange(500, 3000, 100)
target_temperatures = target_temperatures.tolist()
predicted_k_values = {T: [] for T in target_temperatures}

# Iterate through each sheet and perform linear regression on the entire dataset
models = {}
temperature_ranges = {}  # Dictionary to store temperature ranges for each sheet

for sheet in sheet_names:
    # Read and filter the sheet data to remove rows with NaN
    data = excel_data.parse(sheet)
    data = data.dropna(subset=[data.columns[0], data.columns[1]])
    
    # Extract temperature and rate constant values
    temperature = data.iloc[:, 0]
    rate_constant = data.iloc[:, 1]
    
    # Store the temperature range for the current sheet
    temp_min, temp_max = temperature.min(), temperature.max()
    temperature_ranges[sheet] = (temp_min, temp_max)
    
    # Transform the data for linear regression (1000/T and log10(k))
    x = (1000 / temperature).values.reshape(-1, 1)  # Reshape for sklearn compatibility
    y = np.log10(rate_constant).values  # log10(k)
    
    # Create and train the linear regression model using the entire dataset
    model = LinearRegression()
    model.fit(x, y)
    
    # Calculate the model's R2 score on the entire dataset
    y_pred = model.predict(x)
    score = r2_score(y, y_pred)
    print(f"Model score for sheet '{sheet}': R2 = {score:.4f}")
    
    # Save the model with the sheet name
    model_filename = f"model_{sheet}.pkl"
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    
    # Store the trained model for future predictions
    models[sheet] = model

# Calculate and print uncertainty factors for target temperatures
y_values_max = []
for T in target_temperatures:
    relevant_k_values = []

    # Filter datasets based on temperature range
    for sheet, model in models.items():
        temp_min, temp_max = temperature_ranges[sheet]
        
        # Only use models where the target temperature is within the range
        if temp_min <= T <= temp_max:
            # Predict k value for the target temperature T
            x_val = np.array([[1000 / T]])  # Input for the model
            log_k = model.predict(x_val)[0]
            k_value = 10 ** log_k  # Convert back from log10(k) to k
            relevant_k_values.append(k_value)

    # Calculate uncertainty factor only if there are relevant k values
    if relevant_k_values:
        k0 = np.mean(relevant_k_values)   # Mean rate constant
        k_max = np.max(relevant_k_values)  # Maximum rate constant
        uncertainty_factor = np.log10(k_max / k0)  # f(T) calculation
        y_values_max.append(uncertainty_factor)
        print(f"Uncertainty factor f({T} K) = {uncertainty_factor:.4f}")
    else:
        print(f"No relevant models found for f({T} K), as no datasets contain this temperature.")
    
    # Calculate and print uncertainty factors for target temperatures
y_values_min = []
for T in target_temperatures:
    relevant_k_values = []

    # Filter datasets based on temperature range
    for sheet, model in models.items():
        temp_min, temp_max = temperature_ranges[sheet]
        
        # Only use models where the target temperature is within the range
        if temp_min <= T <= temp_max:
            # Predict k value for the target temperature T
            x_val = np.array([[1000 / T]])  # Input for the model
            log_k = model.predict(x_val)[0]
            k_value = 10 ** log_k  # Convert back from log10(k) to k
            relevant_k_values.append(k_value)

    # Calculate uncertainty factor only if there are relevant k values
    if relevant_k_values:
        k0 = np.mean(relevant_k_values)   # Mean rate constant
        k_min = np.min(relevant_k_values)  # Maximum rate constant
        uncertainty_factor = np.log10(k0 / k_min)  # f(T) calculation
        y_values_min.append(uncertainty_factor)
        print(f"Uncertainty factor f({T} K) = {uncertainty_factor:.4f}")
    else:
        print(f"No relevant models found for f({T} K), as no datasets contain this temperature.")

plt.xlabel("Temperature(K)")
plt.ylabel("f(T)")
plt.plot(target_temperatures, y_values_max, label="Using k_max", color="blue", linestyle="-", marker="o")
plt.plot(target_temperatures, y_values_min, label="Using k_min", color="red", linestyle="-", marker="o")
plt.legend()