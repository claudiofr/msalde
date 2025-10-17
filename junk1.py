import pandas as pd

dir="/home/claudiof"

# Sample DataFrame with index
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Score': [1,2,3]
}, index=[102, 103, 101])

# Dictionary to add as a new column
scores = {
    101: 88,
    102: 92,
    103: 85,
    105: 93
}

# Add the dictionary as a new column
df['Score'] = pd.Series(scores)

print(df)