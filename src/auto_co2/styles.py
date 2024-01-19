import numpy as np
import pandas as pd
import os
import datetime
#from datetime import datetime
import io
import re
import plotly.graph_objects as go
from sklearn.metrics import (classification_report, mean_squared_error,
                             mean_absolute_error, r2_score)



########## Project color style ##########
def generate_styles(header_color='steelblue', odd_row_color='aliceblue', even_row_color='white'):
    styles = [
        {'selector': 'th', 'props': [('background-color', header_color), ('color', 'white')]},  # HEADER
        {'selector': 'tr:nth-of-type(odd)', 'props': [('background-color', odd_row_color)]},  # ODD ROWS
        {'selector': 'tr:nth-of-type(even)', 'props': [('background-color', even_row_color)]}  # EVEN ROWS
    ]
    return styles


########## End project style



########## Table Tools ##########
def displayer(df, n=5, styles=None, title=None, save=True):  # Added title parameter
    if styles is None:
        styles = generate_styles()
    
    if isinstance(df, pd.DataFrame):
        format_dict = {col: "{:.3f}" for col in df.select_dtypes('float').columns}
        styled_df = df.head(n).style.format(format_dict).set_table_styles(styles)
    else:  # Assume it's a Styler object
        styled_df = df
    
    if title is not None:  # If a title is provided, set it
        styled_df = styled_df.set_caption(f'<h2>{title}</h2>')  # Use HTML tags to increase the size
    
    if save:
        save_styled_df(styled_df, title)
    
    display(styled_df)
    print(type(styled_df))
    

def display_info(df, styles=None, title=None, save=True): # PANDAS (style)
    if styles is None:
        styles = generate_styles()      
    
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    lines = info_str.split("\n")
    
    # Infos des colonnes
    column_info = lines[5:-3]
    column_df = pd.DataFrame([re.split(r'\s\s+', line.strip()) for line in column_info])
    column_df = column_df.iloc[:, 1:]
    column_names = ['Column', 'Non-Null Count', 'Dtype']
    column_df.columns = column_names[:len(column_df.columns)]
    
    # Infos générales
    general_info = lines[1:3] + lines[-4:-1]
    general_df = pd.DataFrame(general_info, columns=['Info'])
    
    styled_column_df = column_df.style.set_table_styles(styles)
    styled_general_df = general_df.style.set_table_styles(styles)
    
    displayer(column_df, n=len(column_df), styles=styles, title=title, save=save)
    displayer(general_df, n=len(general_df), styles=styles, title=title, save=save)
    
    
def display_describe(df, styles=None, title=None, save=True): # PANDAS (style)
    describe_df = df.describe().transpose().reset_index()
    describe_df.columns = ['Colonne', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    
    displayer(describe_df, n=len(describe_df), styles=styles, save=save, title=title)



def display_na(df, title=None, save=True): # PANSAS (style)
    proportions = df.isna().mean().sort_values(ascending=False)*100
    proportions = proportions.round(2).to_frame().reset_index()
    proportions.columns = ['Colonne', 'Valeurs manquantes (%)']
    
    # Create a styled DataFrame
    styled_df = proportions.style.bar(subset=['Valeurs manquantes (%)'], align='left', color=['lightcoral', 'lightcoral'])
    
    # Set table styles
    styles = [
        {'selector': 'th', 'props': [('background-color', 'steelblue'), ('color', 'white')]},  # Header color
        {'selector': 'tr:nth-of-type(odd)', 'props': [('background-color', 'aliceblue')]},  # Row colors
        {'selector': 'tr:nth-of-type(even)', 'props': [('background-color', 'white')]}
    ]
    styled_df = styled_df.set_table_styles(styles)
    styled_df.format({'Valeurs manquantes (%)': '{:,.2f}%'})
    
    displayer(styled_df, n=len(df), styles=styles, title=title, save=save)

    
def save_styled_df(styled_df, name=None):
    os.makedirs("../output/tables", exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = name if name is not None else 'untitled'
    base_filename = re.sub('[^a-zA-Z0-9-_]', '_', base_filename)
    filename = f"{base_filename}_{timestamp}"
    
    html = styled_df.to_html()
    with open(f"../output/tables/{filename}.html", "w", encoding='utf-8') as f:  # 'utf-8' pour afficher les accents
        f.write(html)
########## End of table tools ##########


########## ML Tables ##########

def display_classification_report(y_true, y_pred, title=None, styles=None): 
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    if styles is None:
        styles = generate_styles()
    
    displayer(report_df, n=len(report_df), styles=styles, title=title)
    
    
def display_feature_importances(model, data, title=None):
    feats = {}
    for feature, importance in zip(data.columns, model.feature_importances_):
        feats[feature] = importance

    importances  = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0:'importance'})
    importances = importances.sort_values(by='importance', ascending=False)
    displayer(importances, n=None, title=title)
    
    
def display_regression_report(actual_values, predicted_values, set_name):
    mean_squared_err = mean_squared_error(actual_values, predicted_values)
    root_mean_squared_err = np.sqrt(mean_squared_err)
    mean_absolute_err = mean_absolute_error(actual_values, predicted_values)
    r2_scoring = r2_score(actual_values, predicted_values)
    
    # Create a DataFrame with the calculated metrics
    report = pd.DataFrame({
        set_name: [mean_squared_err, root_mean_squared_err, mean_absolute_err, r2_scoring]
    }, index=['Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error', 'R2 Score'])
    
    return report

def display_combined_report(y_train, y_pred_train, y_test, y_pred_test, title=None, styles=None):
    # Generate reports for train and test sets
    train_report = display_regression_report(y_train, y_pred_train, 'Train Set')
    test_report = display_regression_report(y_test, y_pred_test, 'Test Set')

    # Combine the reports
    combined_report = pd.concat([train_report, test_report], axis=1)

    # Display the combined report
    displayer(combined_report, n=len(combined_report), styles=styles, title=title)