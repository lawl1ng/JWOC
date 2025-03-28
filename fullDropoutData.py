import os
import re
import string
import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
import matplotlib as mpl
from textblob import TextBlob
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.patches as mpatches

# Load the excel file
df = pd.read_excel("/home/cillian/repos/JWOC_copy/fullDropoutData.xlsx", engine="openpyxl")
df.shape
df.head()
df.tail()

# Remove any rows (none to remove) and columns
df.drop_duplicates(inplace=True)
df.drop(columns=["No"], inplace=True)
df.shape

# Fill missing values with "Unknown"
# df['Reason'].fillna("Unknown", inplace=True)
df.fillna({"Reason":"unknown"}, inplace=True)
df.shape
df.head()

# Drop missing data for Year
# df = df.dropna(subset=['Year'])
# df.shape

# Standardise text data - convert to lower case, remove extra spaces
def clean_text(text):
    if isinstance(text, str):
        text = text.lower().strip() # Convert to lowercase and remove extra spaces
        text = re.sub(r'\s+', ' ', text) # Remove excessive whitespace
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Remove special characters
        text = text.translate(str.maketrans("", "", string.punctuation)) # Remove punctuation
    return(text)

df['Clean Text'] = df['Reason'].apply(clean_text)
df

# Correct spelling while preserving abbreviation of jwoc
def correct_spelling(text, abbreviations):
    if not isinstance(text, str):
        return text

    words = text.split()  # Split into words
    corrected_words = [
        word if word in abbreviations else str(TextBlob(word).correct()) for word in words
    ]

    return " ".join(corrected_words)

df['Clean Text'] = df['Clean Text'].apply(lambda x: correct_spelling(x, {'jwoc','jwocs'}))
df

# Correct spelling of scholorship in Program column and change format of Year column, add backfill of year
df['Program'] = df['Program'].replace({'Scholorship':'Scholarship'})
df.fillna({'Year':'0'}, inplace=True)
df['Year'] = df['Year'].astype(int).astype(str)
df = df.replace({'0':'Missing Year Data'})

# Categorise Dropout Reasons
# Define keyword-to-category mapping with priority order
priority_categories = [
    ("Dismissed/Motivation/No contact/Absence", ['no contact', 'absence', 'cannot reach', 'can contact', 'never come', 'disciplinary', 'misconduct', 'terminated', 'no commitment', 'never show', 'not follow jwoc rules', 'broke the jwocs guideline']),
    ("Personal Health/Family Related Issue", ['health', 'sick', 'illness', 'hospital', 'disease', 'stress', 'accident', 'operation', 'married', 'family', 'parent', 'mother', 'father', 'aunt', 'uncle', 'grandmother']),
    #("Personal/Family Health", ['health', 'sick', 'illness', 'hospital', 'disease', 'stress', 'accident']),
    ("Education", ['scholarship', 'study', 'school', 'university', 'exam', 'test', 'student', 'english course', 'learning']),
    ("Work", ['small business', 'job', 'work', 'internship', 'career', 'offered a position']),
    ("Relocation/Logistics/Other", ['move', 'moved', 'relocation','transportation', 'internet', 'phone', 'will finalise', 'get support from', 'can afford', 'wants to support', 'passed away'])
]


exclusions = {
    "Education": ['no intense to study', 'no time to study', 'no phone to study', 'no willing to study', 'not transportation to study', 'terminated from scholarship', 'interfere his study'],
    "Work": ['lost job', 'looking for work']
}

# Function to categorise reasons
def categorize_reason(reason):
    if reason.lower() == "unknown":
        return "Unknown - Missing Data", None  # Handle unknown cases separately

    main_reason = None
    additional_reasons = set()
    reason_lower = reason.lower()

    for category, keywords in priority_categories:
        for keyword in keywords:
            if keyword in reason_lower:
                if not main_reason:
                    main_reason = category  # First match becomes the main reason
                else:
                    additional_reasons.add(category)  # Store additional categories

    return main_reason or "Other", ', '.join(additional_reasons) if additional_reasons else None

# Apply function to create new columns
df[['Main Reason', 'Additional Reasons']] = df['Clean Text'].apply(lambda x: pd.Series(categorize_reason(x)))

# Save cleaned data to csv and find location
df.to_csv("/home/cillian/repos/JWOC_copy/cleaned_data.csv", index=False)
print(os.getcwd())
df.columns

# Create smaller dataset containing only Year, Summary Reason. Potentially add Program and Additional Reasons later
#dfsub = df[['Year','Program','Sex','Main Reason','Additional Reasons']]
#dfsub.rename(columns={"Main Reason":"Reason"},inplace=True)
df.sort_values(by=['Year','Program','Sex','Main Reason'])

# Plot Years with "Missing Year first"
# Define the categorical order with "Missing Year" first
year_order = sorted(df['Year'].unique(), key=lambda x: (x != "Missing Year Data", x))
year_order
# Count occurrences and reindex
year_counts = df.Year.value_counts()
year_counts = year_counts.reindex(year_order, fill_value=0)
year_counts
# Plot dropouts per year
# df.Year.value_counts()[df.Year.unique()].plot(kind='bar', rot=1)
year_counts.plot(kind='bar', rot=1)
plt.title("Number of Dropouts Per Year", fontsize=24)
plt.xlabel("Year", fontsize=24)
plt.ylabel("Dropouts", fontsize=24)
plt.tick_params(axis="both", labelsize=20)
plt.axvline(x=0.5, color='black', linestyle='dotted', linewidth=2)
plt.show()
missing_percent = round(year_counts['Missing Year Data'] / year_counts.sum() * 100, 2)


# Plot dropouts per group
df.Program.value_counts().plot(kind='bar', rot=1)
plt.title("Number of Dropouts Per Group", fontsize=24)
plt.xlabel("Program", fontsize=24)
plt.ylabel("Dropouts", fontsize=24)
plt.tick_params(axis="both", labelsize=20)
plt.show()

# Plot dropouts per sex
df.Sex.value_counts().plot(kind='bar', rot=1)
plt.title("Number of Dropouts Per Sex", fontsize=24)
plt.xlabel("Sex", fontsize=24)
plt.ylabel("Dropouts", fontsize=24)
plt.tick_params(axis="both", labelsize=20)
plt.show()

# Plot dropouts per year per group
df_counts = df[['Year', 'Program']].value_counts().unstack()
df_counts = df_counts.reindex(year_order, fill_value=0)
#df[['Year', 'Program']].value_counts().unstack().plot(kind='bar', rot=1)
df_counts.plot(kind='bar', rot=1)
plt.title("Number of Dropouts per Group Per Year", fontsize=24)
plt.xlabel("Year", fontsize=24)
plt.ylabel("Dropouts", fontsize=24)
plt.tick_params(axis="both", labelsize=20)
plt.legend(fontsize=24)
plt.axvline(x=0.5, color='black', linestyle='dotted', linewidth=2)
plt.show()

# Plot dropouts per reason
cmap = colormaps.get_cmap('Set2_r') # Get the colormap
value_counts = df['Main Reason'].value_counts() # Value counts for the bar plot
# Define abbreviation mapping for x-axis
abbr_mapping = {
    "Work": "W",
    "Personal Health/Family Related Issue": "PH/FRI",
    "Unknown - Missing Data": "U-MD",
    "Education": "E",
    "Dismissed/Motivation/No contact/Absence": "D/M/NC/A",
    "Relocation/Logistics/Other": "R/L/O"
}
# Define full descriptions for legend
full_mapping = {
    "W": "Work",
    "PH/FRI": "Personal Health/Family Related Issue",
    "U-MD": "Unkown - Missing Data",
    "E": "Education",
    "D/M/NC/A": "Dismissed/Motivation/No contact/Absence",
    "R/L/O": "Relocation/Logistics/Other"
}
# Generate colors using the colormap
colors = [cmap(i / len(value_counts)) for i in range(len(value_counts))]
# Create bar plot
ax = value_counts.plot(kind='bar', color=colors, rot=0)  # rot=0 to keep labels horizontal
# Apply abbreviated x-axis labels
abbreviated_labels = [abbr_mapping.get(label, label) for label in value_counts.index]
ax.set_xticklabels(abbreviated_labels)
# Create legend with full descriptions
legend_patches = [mpatches.Patch(color=colors[i], label=full_mapping[abbr]) for i, abbr in enumerate(abbreviated_labels)]
plt.legend(handles=legend_patches, fontsize=19, bbox_to_anchor=(0.68, 1), loc='upper left', borderaxespad=0)
plt.title("Number of Dropouts Per Reason", fontsize=24)
plt.xlabel("Reason", fontsize=24)
plt.ylabel("Dropouts", fontsize=24)
plt.tick_params(axis="both", labelsize=20)
plt.show()

# Plot dropouts per group per reason
df[['Program', 'Main Reason']].value_counts().unstack().plot(kind='bar', rot=1, color=colors)
plt.title("Number of Dropouts Per Group Per Reason", fontsize=24)
plt.xlabel("Group", fontsize=24)
plt.ylabel("Dropouts", fontsize=24)
plt.tick_params(axis="both", labelsize=20)
plt.legend(fontsize=20)
plt.grid(True, axis='y')
plt.show()

# Plot dropouts per year per reason
df_counts = df[['Year', 'Main Reason']].value_counts().unstack()
df_counts = df_counts.reindex(year_order, fill_value=0)
#df[['Year', 'Program']].value_counts().unstack().plot(kind='bar', rot=1)
df_counts.plot(kind='bar', rot=1)
#df[['Year', 'Main Reason']].value_counts().unstack().plot(kind='bar', rot=1)
plt.title("Number of Dropouts Per Year Per Reason", fontsize=24)
plt.xlabel("Year", fontsize=24)
plt.ylabel("Dropouts", fontsize=24)
plt.tick_params(axis="both", labelsize=20)
plt.legend(fontsize=20)
plt.axvline(x=0.5, color='black', linestyle='dotted', linewidth=2)
plt.grid(True, axis='y')
plt.show()

# Plot dropout per sex per reason
df[['Sex', 'Main Reason']].value_counts().unstack().plot(kind='bar', rot=1)
plt.title("Number of Dropouts Per Sex Per Reason", fontsize=24)
plt.xlabel("Sex", fontsize=24)
plt.ylabel("Dropouts", fontsize=24)
plt.tick_params(axis="both", labelsize=20)
plt.legend(fontsize=20, loc='best')
plt.grid(True, axis='y')
plt.show()


# Count Reasons per year
reason_counts = df.groupby(["Year","Main Reason"]).size().unstack()
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
plt.legend(fontsize=1)
plt.show()