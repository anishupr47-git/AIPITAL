import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#clean the data

def generate_eda_stats(filepath):
    """
    Generate EDA Stats
    """

    #step 1 load data
    print("Starting EDA Process")
    print(f"Reading from file {filepath}")
    df = pd.read_excel(filepath, engine='openpyxl')

    #count rows
    total_rows = df.shape[0]
    print(f"Total number of rows:{total_rows}")

    #count cols
    total_cols = df.shape[1]
    print(f"Total number of columns:{total_cols}")

    #check for missing value
    missing_summary = df.isnull.sum()
    print(f"Missing Values are {missing_summary}")

    #check all column individually
    missing_cols = missing_summary[missing_summary > 0].to_dict()
    print(f"Columns with atleast one missing value{missing_cols}")

    #now we check descriptive value for min,max
    desc_stats = df.describe().to_dict

    target_col= 'Heart_Disease_Diagnosis'
    print(f"Checking if the column {target_col} exists.")
    if target_col in df.columns:
        target_dist = df[target_col].value_counts(normalize=True).to_dict()
        print(f"Target Distribution is:{target_dist}")
    else:
        print("Target Column Not Found")
        target_dist = {}

    print("EDA stats generation complete. Returing dictionary of stats")
    
    return {
        'total_rows': total_rows,
        'total_cols': total_cols,
        'missing_columns': missing_cols,
        'descriptive_stats': desc_stats,
        'target_distribution': target_dist,
        'raw_df': df
    }

def absolute_precprocessing_pipeline(filepath):
    """
    Reads the excel file
    """
    print("Start Preprocessing Pipeline")
    print("Load data into pandas")
    df = pd.read_excel(filepath, open='openpyxl')
    print("Check patient ID")
    if 'Patient_ID' in df.columns:
        print("Found Patient ID")
        df= df.drop(columns=['Patient_ID'])
    else:
        print("Targeted Column is not found")

    #check every empty value to fill them for better training
    print("Looping through column to check types for missing value")

    for col in df.columns:
        print(f"Checking col:{col}")
        col_type=df[col].dtype
        print(f"Type is {col_type}")

        #check for missing value
        missing_count = df[col].isnull().sum()
        print(f"Missing Values are {missing_count}")

        if col_type in ['int64','float64']:
            print("This is numerical type column")
            #calculate median
            median_val = df[col].median()
            print(f"Calculated Median:{median_val}")

            if missing_count > 0:
                print("Filling missing values")
                df[col] = df[col].fillna(median_val)
            else:
                print("No missing value")
        else:
            print("This is a categorical column")
            #calculate mode
            mode_val = df[col].mode()
            print(f"Calculated Mode:{mode_val}")

            if missing_count > 0:
                print("Filling missing value")
                df[col] = df[col].fillna(mode_val)
            else:
                print("No missing value")

    #seperating x and y
    print("Seperating the target variable from the features")
    target_col = 'Heart_Disease_Diagnosis'

    #here target col is must
    if target_col not in df.columns:
        print("ERROR! Target column not found")
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    print(f"Extracting '{target_col}' as our y variable")
    y=df[target_col].values

    print("Dropping target column from x feature")
    X_df = df.drop(columns=[target_col])

    #identifying for column ui
    print("Identifying categorical and numerical columns for the UI")
    cat_cols = X_df.select_dtypes(include=['object','category']).columns.to_list()
    num_cols = X_df.select_dtypes(include=['int64','float64']).columns.to_list()

    print(f"Categorical columns found: {len(cat_cols)}")
    print(cat_cols)
    print(f"Numerical columns found: {len(num_cols)}")
    print(num_cols)
    print("Building Dictionary")
    ui_config={}

    print("Configuring Numerical Columns")
    for col in num_cols:
        print(f"{col}")

        min_val = float(X_df[col].min())
        max_val = float(X_df[col].max())
        mean_val = float(X_df[col].mean())

        if col=='Age':
            print("Number should fall under min max value")
            min_val=1.0
            max_val=150
        else:
            min_val=0.0
            max_val= max_val + 50.0
        
        ui_config[col]= {
            'type':'continious',
            'min':min_val,
            'max':max_val,
            'mean':mean_val
        }

        print("Configuring categorical columns")
        for col in cat_cols:
            print(f"{col}")

            #we need unique
            option_list = X_df[col].unique().tolist()
            mode_val = X_df[col].mode()[0]

            ui_config[col] = {
                'type': 'categorical',
                'options': option_list,
                'mode_val': mode_val
            }

        #we encode now
        #use getdummies
        print("Performing One hot encoding on categorical columns")
        X_encoded = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)

        #we save the final
        feature_names = X_encoded.columns.to_list()
        print(f"Total features after encoding: {len(feature_names)}")

        #scaling
        print("Applying StandardScalers to normalize all")
        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X_encoded.values)

        #return
        return X_scaled, y, scaler, feature_names, ui_config, cat_cols



