import os
import numpy as np
import pandas as pd
import zipfile
import datetime
from xgboost import XGBClassifier, XGBRegressor
import kaggle
import pickle
from tensorflow.keras.models import Model



########## Fetching data from Kaggle ##########

########### Proposition de modification #############################################
# Cela concerne 2 fonctions

def download_co2_data( auth_file_path=None, filepath='../data/raw'):
    dataset_id = 'matthieulenozach/automobile-co2-emissions-eu-2021'
    dataset_name = dataset_id.split('/')[-1]

    if auth_file_path is None:   
        os.system(f'kaggle datasets download -d {dataset_id} -p {filepath}')
    else:
        # Save the original KAGGLE_CONFIG_DIR
        original_kaggle_config_dir = os.environ.get('KAGGLE_CONFIG_DIR')
        # Point KAGGLE_CONFIG_DIR to the directory containing kaggle.json
        os.environ['KAGGLE_CONFIG_DIR'] = os.path.dirname(auth_file_path)

        os.system(f'kaggle datasets download -d {dataset_id} -p {filepath}')
        # Restore the original KAGGLE_CONFIG_DIR
        os.environ['KAGGLE_CONFIG_DIR'] = original_kaggle_config_dir
    return dataset_name

# def download_co2_data(auth_file_path, filepath='../data/raw'):
#     dataset_id = 'matthieulenozach/automobile-co2-emissions-eu-2021'
#     dataset_name = dataset_id.split('/')[-1]  # Get the dataset name

#     # Save the original KAGGLE_CONFIG_DIR
#     original_kaggle_config_dir = os.environ.get('KAGGLE_CONFIG_DIR')
#     # Point KAGGLE_CONFIG_DIR to the directory containing kaggle.json
#     os.environ['KAGGLE_CONFIG_DIR'] = os.path.dirname(auth_file_path)
#     os.system(f'kaggle datasets download -d {dataset_id} -p {filepath}')

#     # Restore the original KAGGLE_CONFIG_DIR
#     if original_kaggle_config_dir is not None:
#         os.environ['KAGGLE_CONFIG_DIR'] = original_kaggle_config_dir
#     else:
#         del os.environ['KAGGLE_CONFIG_DIR']
#     return dataset_name


# Fonction non modifiee
def load_co2_data(dataset_name="automobile-co2-emissions-eu-2021", filepath='../data/raw'):
    # read zip file and extract in {filepath}
    with zipfile.ZipFile(f'{filepath}/{dataset_name}.zip', 'r') as zip_ref:
        zip_ref.extractall(filepath)
    filename = zip_ref.namelist()[0] 
    data = pd.read_csv(f'{filepath}/{filename}', low_memory=False)
    return data

# Fonction Modifiée
def download_and_load_co2_data( auth_file_path=None, filepath='../data/raw'):
    dataset_name = download_co2_data(auth_file_path, filepath)
    data = load_co2_data(dataset_name, filepath)
    return data

# def download_and_load_co2_data(auth_file_path, filepath='../data/raw'):
#     dataset_name = download_co2_data(auth_file_path, filepath)
#     data = load_co2_data(dataset_name, filepath)
#     return data

########### Fin de modification #############################################

 
########## End of fetching data from Kaggle ##########


########## General data cleaning ##########
def convert_dtypes(df):
    floats = df.select_dtypes(include=['float']).columns
    df.loc[:, floats] = df.loc[:, floats].astype('float32')

    ints = df.select_dtypes(include=['int']).columns
    df.loc[:, ints] = df.loc[:, ints].astype('float32')

    objs = df.select_dtypes(include=['object']).columns
    df.loc[:, objs] = df.loc[:, objs].astype('category')

    return df


def rename_columns(df):
    abbrev_list = [
        'ID', 'Country', 'VFN', 'Mp', 'Mh', 'Man', 'MMS', 'Tan', 'T', 'Va', 'Ve', 'Mk',
        'Cn', 'Ct', 'Cr', 'r', 'm (kg)', 'Mt', 'Enedc (g/km)', 'Ewltp (g/km)', 'W (mm)',
        'At1 (mm)', 'At2 (mm)', 'Ft', 'Fm', 'ec (cm3)', 'ep (KW)', 'z (Wh/km)', 'IT',
        'Ernedc (g/km)', 'Erwltp (g/km)', 'De', 'Vf', 'Status', 'year', 'Date of registration',
        'Fuel consumption ', 'Electric range (km)'
        ]

    nom_colonne_list = [
        'ID', 'Country', 'VehicleFamilyIdentification', 'Pool', 'ManufacturerName', 'ManufNameOem',
        'ManufNameMS', 'TypeApprovalNumber', 'Type', 'Variant', 'Version', 'Make', 'CommercialName',
        'VehicleCategory', 'CategoryOf', 'TotalNewRegistrations', 'MassRunningOrder',
        'WltpTestMass', 'Co2EmissionsNedc', 'Co2EmissionsWltp',
        'BaseWheel', 'AxleWidthSteering', 'AxleWidthOther', 'FuelType', 'FuelMode',
        'EngineCapacity', 'EnginePower', 'ElectricConsumption',
        'InnovativeTechnology', 'InnovativeEmissionsReduction',
        'InnovativeEmissionsReductionWltp', 'DeviationFactor', 'VerificationFactor',
        'Status', 'RegistrationYear', 'RegistrationDate', 'FuelConsumption', 'ElectricRange'
        ]
    
    name_dict = dict(zip(abbrev_list, nom_colonne_list))
    df = df.rename(columns = name_dict)
    return df

def select_countries(df, countries:list):
    return df[df['Country'].isin(countries)]


def drop_irrelevant_columns(df):
    to_drop = [
            'VehicleFamilyIdentification', 'ManufNameMS', 'TypeApprovalNumber', 
            'Type', 'Variant', 'Version', 'VehicleCategory',
           'TotalNewRegistrations', 'Co2EmissionsNedc', 'AxleWidthOther', 'Status',
           'InnovativeEmissionsReduction', 'DeviationFactor', 'VerificationFactor', 'Status',
           'RegistrationYear']
    
    df = df.drop(columns=[col for col in to_drop if col in df.columns])
    return df

def conditional_column_update(df, condition_column, condition_value, target_column, target_value):
    df.loc[df[condition_column] == condition_value, target_column] = target_value
    return df

def clean_manufacturer_columns(df): # VIZ
    df = conditional_column_update(df, 'Make', 'TESLA', 'Pool', 'TESLA')
    df = conditional_column_update(df, 'Make', 'HONDA', 'Pool', 'HONDA-GROUP')
    df = conditional_column_update(df, 'Make', 'JAGUAR', 'Pool', 'TATA-MOTORS')
    df = conditional_column_update(df, 'Make', 'LAND ROVER', 'Pool', 'TATA-MOTORS')
    df = conditional_column_update(df, 'Make', 'RANGE-ROVER', 'Pool', 'TATA-MOTORS')

    df['Make'] = df['Make'].str.upper()
    df = conditional_column_update(df, 'Make', 'B M W', 'Make', 'BMW')
    df = conditional_column_update(df, 'Make', 'BMW I', 'Make', 'BMW')
    df = conditional_column_update(df, 'Make', 'ROLLS ROYCE I', 'Make', 'ROLLS-ROYCE')
    df = conditional_column_update(df, 'Make', 'FORD (D)', 'Make', 'FORD') 
    df = conditional_column_update(df, 'Make', 'FORD-CNG-TECHNIK', 'Make', 'FORD')
    df = conditional_column_update(df, 'Make', 'FORD - CNG-TECHNIK', 'Make', 'FORD')
    df = conditional_column_update(df, 'Make', 'MERCEDES', 'Make', 'MERCEDES-BENZ')
    df = conditional_column_update(df, 'Make', 'MERCEDES BENZ', 'Make', 'MERCEDES-BENZ') 
    df = conditional_column_update(df, 'Make', 'LYNK & CO', 'Make', 'LYNK&CO')  
    df = conditional_column_update(df, 'Make', 'VOLKSWAGEN VW', 'Make', 'VOLKSWAGEN')
    df = conditional_column_update(df, 'Make', 'VOLKSWAGEN, VW', 'Make', 'VOLKSWAGEN') 
    df = conditional_column_update(df, 'Make', 'VOLKSWAGEN. VW	', 'Make', 'VOLKSWAGEN') 
    df = conditional_column_update(df, 'Make', 'VW', 'Make', 'VOLKSWAGEN')
    return df

def correct_fueltype(df): # VIZ
    df.loc[df['FuelType'] == 'petrol/electric', 'FuelType'] = 'PETROL/ELECTRIC'
    df.loc[df['FuelType'] == 'E85', 'FuelType'] = 'ETHANOL'
    df.loc[df['FuelType'] == 'NG-BIOMETHANE', 'FuelType'] = 'NATURALGAS'
    return df

    # ***** High Level Function: PRE CLEANING ***** #
def data_preprocess(df, countries=None):
    if countries is not None:
        df = select_countries(df, countries)
    df = convert_dtypes(df)
    df = rename_columns(df)
    return df
    # ***** End of High Level Function: PRE CLEANING ***** #

    # ***** High Level Function: VIZ CLEANING ***** #
def dataviz_preprocess(df, countries=None):
    df = data_preprocess(df, countries)
    df = drop_irrelevant_columns(df)
    df = clean_manufacturer_columns(df)
    df = correct_fueltype(df)
    df = discretize_co2(df)
    df.loc[df['ElectricRange'].isna(), 'ElectricRange'] = 0 
    df = drop_residual_incomplete_rows(df)  
    df['CommercialName'] = df['CommercialName'].str.replace('[^a-zA-Z0-9\s/-]', '')
    return df
    # ***** End of High Level Function: VIZ ***** #

########## End of General Data Cleaning ##########


########## ML Preprocessing ##########

def remove_columns(df, columns=None, axlewidth=True, engine_capacity=True, fuel_consumption=True):
    columns_to_drop = ['Country', 'ManufacturerName', 'ManufNameOem', 'Type', 'Variant', 
                        'Version', 'Make', 'CommercialName', 'VehicleCategory', 'CommercialName',
                        'TotalNewRegistrations', 'Co2EmissionsNedc', 'WltpTestMass','FuelMode', 
                        'ElectricConsumption', 'InnovativeEmissionsReduction', 'DeviationFactor', 
                        'VerificationFactor', 'Status','RegistrationYear', 'RegistrationDate',
                        'CategoryOf', 'InnovativeEmissionsReductionWltp', 'ID']
    if axlewidth:
        columns_to_drop.append('AxleWidthSteering')
    if engine_capacity:
        columns_to_drop.append('EngineCapacity')
    if fuel_consumption:
        columns_to_drop.append('FuelConsumption')
        
    if columns is None:
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    else:
        df = df.drop(columns=[col for col in columns if col in df.columns])
    return df


def remove_fueltype(df, keep_fossil=False):
    rows_t0 = len(df)
    
    if keep_fossil:
        df.drop(df.loc[df['FuelType'].isin(['HYDROGEN', 'ELECTRIC', 'UNKNOWN'])].index, inplace=True)

    else:
        df = df.drop(columns=['FuelType'])
        
    rows_t1 = len(df)
    print(f"FuelType rows dropped:{rows_t0 - rows_t1}")
    
    return df


def standardize_innovtech(df, drop=False):
    if drop:
        df = df.drop(columns=['InnovativeTechnology'])
    else:
        df['InnovativeTechnology'].fillna(0, inplace=True)
        df.loc[df['InnovativeTechnology'] != 0, 'InnovativeTechnology'] = 1
        df['InnovativeTechnology'] = df['InnovativeTechnology'].astype(int)
    return df
    
    
def drop_residual_incomplete_rows(df):
    rows_t0 = len(df)
    for col in df.columns:
        if df[col].isna().mean() <= 0.05:
            df = df[df[col].notna()]

    rows_t1 = len(df)
    print(f"Incomplete rows dropped:{rows_t0 - rows_t1}")
    return df


# ***** High Level Function: ML CLEANING ***** #
def ml_preprocess(df, countries=None, 
                     rem_fuel_consumption=True, 
                     rem_axlewidth=True, 
                     rem_engine_capacity=True):
    
    rows_t0 = len(df)
    
    df = data_preprocess(df, countries)
    df = drop_irrelevant_columns(df)
    df = remove_columns(df, axlewidth=rem_axlewidth, engine_capacity=rem_engine_capacity, fuel_consumption=rem_fuel_consumption)  
    df = standardize_innovtech(df)
    df['ElectricRange'].fillna(0, inplace=True)
    df = drop_residual_incomplete_rows(df)
 

    rows_t1 = len(df)
    print(f"TOTAL NUMBER OF ROWS DROPPED:{rows_t0 - rows_t1}")
    
    return df
# ***** end of High Level Function: ML CLEANING ***** #

########## End of ML Preprocessing ##########



########## Feature Engineering ##########

def discretize_co2(df):
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    bins = [-1, 100, 120, 140, 160, 200, 250, 1000]

    grades = pd.cut(df['Co2EmissionsWltp'], bins=bins, labels=labels)
    df['Co2Grade'] = grades
    return df    


def discretize_electricrange(df):
    bins = [-float('inf'),0,50,100,150,300]
    labels = ['NO_RANGE', '0to50', '50to100', '100to150', '150+']
    df['ElectricRange'] = pd.cut(df['ElectricRange'], bins=bins, labels=labels)
    return df


def get_classification_data(df):
    df = discretize_co2(df)
    df = df.drop(columns=['Co2EmissionsWltp'])
    return df



def dummify(df, column):
    dummies = pd.get_dummies(data=df[column], dtype=int, prefix=f"{column}")
    df = df.join(dummies)
    df.drop(columns=[column], inplace=True)
    return df

def dummify_all_categoricals(df, dummy_columns=None, should_discretize_electricrange=True):
    if dummy_columns is None:
        df= pd.get_dummies(data = df)
    else:
        for column in dummy_columns:
            df = dummify(df, column)
    return df

# Fonction dummify inutile ?
    
########## End of Feature Engineering ##########



########## Persistence ##########


def save_processed_data(df, classification=False, pickle=True):
    filepath = '../data/processed'
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(filepath, exist_ok=True)
    if classification:
        filename = f'co2_classification_{timestamp}'
        df = get_classification_data(df)
    else:
        filename = f'co2_regression_{timestamp}'
    if pickle:
        df.to_pickle(f"{filepath}/{filename.split('.')[0]}.pkl")
    else:
        df.to_csv(, index=False)
    print(f"Data saved to {filepath}/{filename}")




def save_model(model, model_type='other'):
    model_name = type(model).__name__
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = '../models'
    filename = f"{model_name}_{timestamp}"
    os.makedirs(filepath, exist_ok=True)
    
    if model_type == 'xgb' and isinstance(model, (XGBClassifier, XGBRegressor)):
        filename += '.model'
        full_path = os.path.join(filepath, filename)
        model.save_model(full_path)
    elif model_type == 'keras' and isinstance(model, Model):
        filename += '.h5'
        full_path = os.path.join(filepath, filename)
        model.save(full_path)
    else:
        filename += '.pkl'
        full_path = os.path.join(filepath, filename)
        with open(full_path, 'wb') as f:
            pickle.dump(model, f)
    print(f'Model saved at {full_path}')
    
    
    
    
def save_shap_values(shap_values, shap_sample):
    filename_prefix = type(shap_values).__name__
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = '../output/interpretability'
    filename_shap = f"{filename_prefix}_shap_values_{timestamp}.csv"
    filename_explainer = f"{filename_prefix}_explainer_{timestamp}.csv"
    os.makedirs(filepath, exist_ok=True)
    
    full_path_shap = os.path.join(filepath, filename_shap)
    full_path_explainer = os.path.join(filepath, filename_explainer)
    
    # Save SHAP values to CSV
    shap_values_np = np.concatenate(shap_values, axis=0) if isinstance(shap_values, list) else np.array(shap_values)
    shap_values_df = pd.DataFrame(shap_values_np, columns=shap_sample.columns)
    shap_values_df.to_csv(full_path_shap, index=False)
    print(f'SHAP values saved at {full_path_shap}')

    # Save explainer information to CSV
    with open(full_path_explainer, 'w') as explainer_file:
        explainer_file.write("feature\n")
        explainer_file.write("\n".join(map(str, shap_sample.columns)))
    print(f'Explainer information saved at {full_path_explainer}')