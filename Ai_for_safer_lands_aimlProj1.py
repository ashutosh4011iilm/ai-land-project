
# === Imports (run this first) ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, roc_curve
import joblib
sns.set(style='whitegrid')



# === Load dataset (upload from your computer in Colab) ===
# Option A: Use the Colab file upload dialog
try:
    from google.colab import files
    print('Please upload your CSV file using the file picker.')
    uploaded = files.upload()  # choose the CSV file
    # take the first uploaded filename
    filename = next(iter(uploaded.keys()))
    df = pd.read_csv(filename, encoding='latin1', low_memory=False)
except Exception as e:
    print('Upload failed or not running in Colab — trying local path /mnt/data/...')
    # If you already placed the file in the environment (like here), change the path below:
    filename = '/mnt/data/Detail of Encroachment on State Land.csv'
    df = pd.read_csv(filename, encoding='latin1', low_memory=False)

print('Loaded file:', filename)
print('Rows, columns:', df.shape)
df.head(6)



# === Quick EDA (easy to read) ===
print('\n--- INFO ---')
print(df.info())
print('\n--- DESCRIPTION (numeric) ---')
print(df.describe().transpose().head(20))
print('\n--- SOME COLUMNS ---\n', df.columns.tolist())

# Missing values summary
missing = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(2)
miss_df = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
print('\n--- Missing values (top 20) ---')
print(miss_df.head(20))



# === Simple cleaning (beginner-friendly) ===
# 1) Drop obvious ID column if present
for id_name in ['Sr. No.', 'Sr No', 'SrNo', 'ID', 'Id']:
    if id_name in df.columns:
        df = df.drop(columns=[id_name])
        print('Dropped ID-like column:', id_name)

# 2) Drop columns with more than 50% missing values
thresh = len(df) * 0.5
to_drop = [c for c in df.columns if df[c].isnull().sum() > thresh]
if to_drop:
    print('Dropping columns with >50% missing:', to_drop)
    df = df.drop(columns=to_drop)

# 3) Trim whitespace in string columns
for c in df.select_dtypes(include='object').columns:
    df[c] = df[c].str.strip()

print('After cleaning, shape:', df.shape)
df.head(4)



# === Choose your target column ===
# By default the notebook will pick the last column as target. You can change TARGET below.
# Example: TARGET = 'Encroachment'  (set it to the real column name from df.columns)
TARGET = None  # <-- set a column name as a string if you want to override

if TARGET is None:
    TARGET = df.columns[-1]  # default: last column
print('Using target column:', TARGET)
y = df[TARGET]
X = df.drop(columns=[TARGET])
print('\nTarget unique values (up to 50):', y.unique()[:50])



# === Very simple preprocessing (no fancy pipeline) ===
# Separate numeric and categorical
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

print('Numeric cols:', num_cols)
print('Categorical cols:', cat_cols)

# Fill missing values: median for numeric, 'missing' for categorical
X[num_cols] = X[num_cols].fillna(X[num_cols].median())
X[cat_cols] = X[cat_cols].fillna('missing')

# One-hot encode categorical (simple and safe for beginners)
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
print('After one-hot, feature count:', X.shape[1])



# === Train/test split and model selection (auto-detect classification/regression) ===
# Decide task type: classification if target is string or has <=20 unique values
is_classification = False
if y.dtype == 'O' or y.dtype.name == 'category' or y.nunique() <= 20:
    is_classification = True

print('Task type:', 'Classification' if is_classification else 'Regression')

# If classification and target is string, encode labels
if is_classification:
    y_enc = y.astype(str).factorize()[0]
else:
    y_enc = y.astype(float).values

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
print('Train / test shapes:', X_train.shape, X_test.shape)

# choose a simple model
if is_classification:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)

# train
model.fit(X_train, y_train)
print('Model trained.')



# === Evaluation (simple prints) ===
y_pred = model.predict(X_test)

if is_classification:
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print('\nClassification metrics:')
    print('Accuracy:', round(acc,4))
    print('Precision (weighted):', round(prec,4))
    print('Recall (weighted):', round(rec,4))
    print('F1 (weighted):', round(f1,4))
    print('\nClassification report:\n', classification_report(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)
else:
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print('\nRegression metrics:')
    print('RMSE:', round(rmse,4))
    print('MAE:', round(mae,4))
    print('R2:', round(r2,4))
    cm = None



# === Simple plots (matplotlib + seaborn) ===
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8,5)

# Target distribution
plt.figure()
if is_classification:
    vals = pd.Series(y).astype(str).value_counts()
    vals.plot(kind='bar')
    plt.title('Target class distribution')
else:
    plt.hist(y.dropna(), bins=30)
    plt.title('Target distribution')
plt.show()

# Missing values bar (top 20)
plt.figure(figsize=(8,6))
miss = df.isnull().sum()
miss = miss[miss > 0].sort_values(ascending=False).head(20)
if len(miss) > 0:
    miss.plot(kind='barh')
    plt.title('Top missing values by column')
    plt.gca().invert_yaxis()
else:
    plt.text(0.1, 0.5, 'No missing values found', fontsize=14)
plt.show()

# Confusion matrix heatmap (if classification)
if is_classification and cm is not None:
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()

# Feature importances (top 20) for tree models
try:
    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
    plt.figure(figsize=(8,6))
    fi[::-1].plot(kind='barh')
    plt.title('Top 20 feature importances')
    plt.show()
except Exception as e:
    print('Could not compute feature importances:', e)

# ROC curve if binary classification
if is_classification and len(np.unique(y_train)) == 2:
    try:
        probs = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        plt.plot([0,1], [0,1], '--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()
    except Exception as e:
        print('ROC plot failed:', e)



# === Save model (optional) ===
model_filename = 'encroachment_model.joblib'
joblib.dump(model, model_filename)
print('Saved trained model to', model_filename)

# You can download it from Colab after running this cell.
