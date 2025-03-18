import os
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns


# Load the excel file
df = pd.read_excel("repos/JWOC/dropoutDataJWOC.xlsx", engine="openpyxl")

# Basic EDA
print(df.head()) # Display the first few rows
print(df.info()) # Summary of dataset
print(df.describe()) # Statistics of numerical columns
print(df.columns) # Column names

# Data cleaning
df = df.fillna("No") # Change na values to No
df.rename(columns={"Dropout Reason":"Reason"}, inplace=True) # Rename columns
df.Year = pd.to_datetime(df.Year.astype(str), format='%Y') # Convert Year data type from int64 to Year
df["Summary Reason"].value_counts() # Apply functions to get count of each reason - Compute group sizes / counts
d = {"Family Member Health":"Family Related Issue", "Family work":"Family Related Issue", "Family issue":"Family Related Issue", "Transport Issue":"Other", "Debt":"Other", "Internet/Phone Issue":"Other"} # Create dictionary of replacements
df = df.replace(d) # Replacing values for wider groupings
df.Year = df.Year.dt.strftime('%Y')
print(df)


# Create smaller dataset containing only Year, Summary Reason. Potentially add Program and Additional Reasons later
df2 = df[["Program", "Year", "Summary Reason"]]
df2.rename(columns={"Summary Reason":"Reason"},inplace=True)
df2.sort_values(by=['Year'])

# Plot dropouts per year
df2.Year.value_counts()[df2.Year.unique()].plot(kind='bar', rot=1)
plt.title("Number of Dropouts Per Year", fontsize=24)
plt.xlabel("Year", fontsize=24)
plt.ylabel("Dropouts", fontsize=24)
plt.tick_params(axis="both", labelsize=20)
plt.show()

# Plot dropouts per group
df2.Program.value_counts().plot(kind='bar', rot=1)
plt.title("Number of Dropouts Per Group", fontsize=24)
plt.xlabel("Program", fontsize=24)
plt.ylabel("Dropouts", fontsize=24)
plt.tick_params(axis="both", labelsize=20)
plt.show()

# Plot dropouts per year per group
df2[['Year', 'Program']].value_counts().unstack().plot(kind='bar', rot=1)
plt.title("Number of Dropouts per Group Per Year", fontsize=24)
plt.xlabel("Year", fontsize=24)
plt.ylabel("Dropouts", fontsize=24)
plt.tick_params(axis="both", labelsize=20)
plt.legend(fontsize=24)
plt.show()

# Plot dropouts per group per reason
df2[['Program', 'Reason']].value_counts().unstack().plot(kind='bar', rot=1)
plt.title("Number of Dropouts Per Group Per Reason", fontsize=24)
plt.xlabel("Group", fontsize=24)
plt.ylabel("Dropouts", fontsize=24)
plt.tick_params(axis="both", labelsize=20)
plt.legend(fontsize=20)
plt.show()

# Plot dropouts per year per reason
df2[['Year', 'Reason']].value_counts().unstack().plot(kind='bar', rot=1)
plt.title("Number of Dropouts Per Year Per Reason", fontsize=24)
plt.xlabel("Year", fontsize=24)
plt.ylabel("Dropouts", fontsize=24)
plt.tick_params(axis="both", labelsize=20)
plt.legend(fontsize=20)
plt.show()

# Count Reasons per year
reason_counts = df2.groupby(["Year","Reason"]).size().unstack()
reason_counts = reason_counts.fillna(0)
print(reason_counts)

# Plot trends over time
reason_counts.plot(kind='line', marker='o', figsize=(12,6))
plt.title("Trends of Reasons Over Time", fontsize=24)
plt.xlabel("Year", fontsize=24)
plt.ylabel("Dropout", fontsize=24)
plt.tick_params(axis="both", labelsize=20)
plt.legend(fontsize=16)
plt.grid(True)
plt.show()

# Stacked Bar Chart
reason_counts.plot(kind='bar', stacked='True', figsize=(12, 6), colormap='viridis', rot=0)
plt.title("Distribution of Reasons by Year", fontsize=24)
plt.xlabel("Year", fontsize=24)
plt.ylabel("Dropout", fontsize=24)
plt.tick_params(axis="both", labelsize=20)
plt.legend(fontsize=14)
plt.show()