from scipy import optimize
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.tree import export_graphviz
from io import StringIO # Standard import for string handling
import matplotlib.pyplot as plt
import seaborn as sns
import pydotplus
# import graphviz # pydotplus handles Graphviz integration

# ==============================================================================
## 1. Data Loading and Initial Analysis
# ==============================================================================

# Read the CSV file
df = pd.read_csv('diabetes.csv')

print(df.describe())

# Function to check the count of values less than a specified value in a column
def chkColumnForVal(col_name, val):
    print(col_name)
    out_array = []
    for index, t in enumerate(df[col_name]):
        if (t < val):
            out_array.append(index)
    return len(out_array)

# Function to calculate Mean and Mode
def cal_mmm(col_name):
    mean = df[col_name].mean()
    mode = df[col_name].mode()
    mmm_array = [mean, mode]
    return mmm_array

# Re-read the data
df = pd.read_csv('diabetes.csv', header=0, sep=',', index_col=None)
print(df.head(5))

# ==============================================================================
## 2. Data Cleaning - Zero Value Replacement
# ==============================================================================

# Replace zero values with the mean for columns where zero is not meaningful (Zero Replacement)
df['Glucose'] = df.Glucose.mask(df.Glucose == 0, cal_mmm("Glucose")[0])
df['BloodPressure'] = df.BloodPressure.mask(df.BloodPressure == 0, cal_mmm("BloodPressure")[0])
df['SkinThickness'] = df.SkinThickness.mask(df.SkinThickness == 0, cal_mmm("SkinThickness")[0])
df['Insulin'] = df.Insulin.mask(df.Insulin == 0, cal_mmm("Insulin")[0])
df['BMI'] = df.BMI.mask(df.BMI == 0, cal_mmm("BMI")[0])
df['DiabetesPedigreeFunction'] = df.DiabetesPedigreeFunction.mask(df.DiabetesPedigreeFunction == 0, cal_mmm("DiabetesPedigreeFunction")[0])
print(df.head(5))

# ==============================================================================
## 3. Data Visualization and Outlier Removal
# ==============================================================================

# Plot Histograms
df.hist(figsize=(10, 8))
plt.show()

# Plot Box plots for outlier detection
df.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False, figsize=(10, 8))
plt.show()

# Outlier removal using the Quantile method (10th and 90th percentiles) for SkinThickness and Insulin
filt_df = df[['SkinThickness', 'Insulin']]
low = .1
high = .9
quant_df = filt_df.quantile([low, high])
print("Quantiles (10% and 90%):\n", quant_df)

# Apply filtering
filt_df = filt_df.apply(lambda x: x[(x > quant_df.loc[low, x.name]) & (x < quant_df.loc[high, x.name])], axis=0)

print("*******after outlier removal*********")

# Update the original DataFrame with filtered data
df['SkinThickness'] = filt_df['SkinThickness']
df['Insulin'] = filt_df['Insulin']

# Remove rows containing NaN values resulting from outlier removal
df.dropna(axis=0, how='any', inplace=True)

print(df.describe())

# Plot Box plots after outlier removal
df.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False, figsize=(10, 8))
plt.show()

# ==============================================================================
## 4. Data Splitting and Model Training
# ==============================================================================

# Split the dataset: Training (60%) and Testing (40%)
train, test = train_test_split(df, test_size=0.4, random_state=30)
target = train["Outcome"]
feature = train[train.columns[0:8]]
feat_names = train.columns[0:8]
target_classes = ['0', '1']
print("\nTest Data Head:\n", test.head())

### **4.1. Decision Tree Model**
model = DecisionTreeClassifier(max_depth=4, random_state=0)
model.fit(feature, target)

test_input = test[test.columns[0:8]]
expected = test["Outcome"]
predicted = model.predict(test_input)

# Decision Tree Evaluation
print("\n--- Decision Tree Classification Report ---")
print(metrics.classification_report(expected, predicted))
conf = metrics.confusion_matrix(expected, predicted)
print("Decision Tree Confusion Matrix:\n", conf)
dtreescore = model.score(test_input, expected)
print("Decision Tree accuracy: ", dtreescore)

# Plot Confusion Matrix
label = ["0", "1"]
sns.heatmap(conf, annot=True, xticklabels=label, yticklabels=label, fmt='d')
plt.title('Decision Tree Confusion Matrix')
plt.show()

# Feature Importance
importance = model.feature_importances_
indices = np.argsort(importance)[::-1]
print("\nDecisionTree Feature ranking:")
for f in range(feature.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, feat_names[indices[f]], importance[indices[f]]))

# Plot Feature Importance 
plt.figure(figsize=(15, 5))
plt.title("DecisionTree Feature importances")
plt.bar(range(feature.shape[1]), importance[indices], color="y", align="center")
plt.xticks(range(feature.shape[1]), feat_names[indices], rotation=45)
plt.xlim([-1, feature.shape[1]])
plt.tight_layout()
plt.show()

### **4.2. K-Nearest Neighbors (KNN) Model**
neigh = KNeighborsClassifier(n_neighbors=21)
neigh.fit(feature, target)
knnpredicted = neigh.predict(test_input)

# KNN Evaluation
print("\n--- KNN Classification Report ---")
print(metrics.classification_report(expected, knnpredicted))
kconf = metrics.confusion_matrix(expected, knnpredicted)
print("KNN Confusion Matrix:\n", kconf)
knnscore = neigh.score(test_input, expected)
print("KNN accuracy: ", knnscore)

# ==============================================================================
## 5. Result Comparison and Tree Visualization
# ==============================================================================

# Plot Performance Comparison (Box Plot)
names_ = ["DT", "KNN"]
results_ = [dtreescore, knnscore]
res = pd.DataFrame({'x': names_, 'y': results_})
sns.boxplot(x='x', y='y', data=res)
plt.title('Model Accuracy Comparison')
plt.show()

# Export and Plot the Decision Tree (Requires Graphviz installation)
dot_data = export_graphviz(model, out_file=None, feature_names=feat_names, class_names=target_classes,
                           filled=True, rounded=True, special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)
print("\nDOT Graph Data (start):\n", dot_data[:500])
# Save the tree as a PNG file
try:
    # Graphviz must be installed and added to PATH
    graph.write_png("diabetes_dtree.png")
    print("\nDecision Tree saved as diabetes_dtree.png")
except Exception as e:
    print(f"\nCould not save Decision Tree PNG: {e}")

# ==============================================================================
## 6. ROC Curves and Performance Evaluation
# ==============================================================================

# ROC Curve for Decision Tree 
fpr, tpr, thres = roc_curve(expected, predicted)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.title('DecisionTreeClassifier - Receiver Operating Characteristic Test Data')
plt.plot(fpr, tpr, color='green', lw=2, label='DecisionTree ROC curve (area = %0.2f)' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# ROC Curve for KNN 
kfpr, ktpr, kthres = roc_curve(expected, knnpredicted)
kroc_auc = auc(kfpr, ktpr)
plt.figure(figsize=(8, 6))
plt.title('KNeighborsClassifier - Receiver Operating Characteristic')
plt.plot(kfpr, ktpr, color='darkorange', lw=2, label='KNeighbors ROC curve (area = %0.2f)' % kroc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()