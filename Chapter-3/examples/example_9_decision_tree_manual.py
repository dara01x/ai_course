"""
Example 9 - Decision Tree Manual Calculation (Tennis Dataset)
This example demonstrates the manual calculation of information gain for decision tree construction
"""

import numpy as np
import pandas as pd

print("=== Decision Tree Manual Calculation - Tennis Dataset ===")

# Create the tennis dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 
                'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 
                   'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 
                'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Windy': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 
             'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 
                  'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)
print("Tennis Dataset:")
print(df)

# Calculate entropy for the target variable
def calculate_entropy(target_col):
    """Calculate entropy of target column"""
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = -np.sum([(counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) 
                      for i in range(len(elements))])
    return entropy

# Calculate information gain
def info_gain(data, split_attribute_name, target_name="PlayTennis"):
    """Calculate information gain for a given attribute"""
    # Calculate entropy of target
    total_entropy = calculate_entropy(data[target_name])
    
    # Calculate weighted entropy for the attribute
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * 
                              calculate_entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) 
                              for i in range(len(vals))])
    
    # Calculate Information Gain
    information_gain = total_entropy - weighted_entropy
    return information_gain

print(f"\n=== Manual Entropy Calculations ===")

# Count target classes
yes_count = (df['PlayTennis'] == 'Yes').sum()
no_count = (df['PlayTennis'] == 'No').sum()
total = len(df)

print(f"Total samples: {total}")
print(f"Yes: {yes_count}, No: {no_count}")

# Calculate main entropy
main_entropy = calculate_entropy(df['PlayTennis'])
print(f"\nEntropy of PlayTennis: {main_entropy:.3f}")

print(f"\n=== Information Gain Calculations ===")

# Calculate information gain for each attribute
attributes = ['Outlook', 'Temperature', 'Humidity', 'Windy']
gains = {}

for attr in attributes:
    gain = info_gain(df, attr)
    gains[attr] = gain
    print(f"Information Gain for {attr}: {gain:.3f}")

# Find the best attribute (highest information gain)
best_attribute = max(gains, key=gains.get)
print(f"\nBest attribute to split on: {best_attribute} (IG = {gains[best_attribute]:.3f})")

print(f"\n=== Detailed Breakdown for Outlook ===")
outlook_counts = df.groupby(['Outlook', 'PlayTennis']).size().unstack(fill_value=0)
print("Outlook breakdown:")
print(outlook_counts)

# Manual calculation for Outlook
sunny_data = df[df['Outlook'] == 'Sunny']
overcast_data = df[df['Outlook'] == 'Overcast']
rain_data = df[df['Outlook'] == 'Rain']

print(f"\nSunny subset: {len(sunny_data)} samples")
print(f"Yes: {(sunny_data['PlayTennis'] == 'Yes').sum()}, No: {(sunny_data['PlayTennis'] == 'No').sum()}")
print(f"Entropy: {calculate_entropy(sunny_data['PlayTennis']):.3f}")

print(f"\nOvercast subset: {len(overcast_data)} samples")
print(f"Yes: {(overcast_data['PlayTennis'] == 'Yes').sum()}, No: {(overcast_data['PlayTennis'] == 'No').sum()}")
print(f"Entropy: {calculate_entropy(overcast_data['PlayTennis']):.3f}")

print(f"\nRain subset: {len(rain_data)} samples")
print(f"Yes: {(rain_data['PlayTennis'] == 'Yes').sum()}, No: {(rain_data['PlayTennis'] == 'No').sum()}")
print(f"Entropy: {calculate_entropy(rain_data['PlayTennis']):.3f}")

print(f"\n=== Decision Tree Construction Process ===")
print("1. Calculate entropy of target variable")
print("2. For each attribute, calculate weighted entropy of subsets")
print("3. Calculate Information Gain = Original Entropy - Weighted Entropy")
print("4. Choose attribute with highest Information Gain as root node")
print("5. Repeat process for each branch until entropy ≈ 0")

print(f"\n=== Key Insights ===")
print(f"• Outlook has the highest information gain ({gains['Outlook']:.3f})")
print("• Overcast always leads to 'Yes' (pure subset)")
print("• Sunny and Rain subsets need further splitting")
print("• This forms the structure of our decision tree")
