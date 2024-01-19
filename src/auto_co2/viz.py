from IPython.display import Image
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots as sp
from plotly.subplots import make_subplots
import colorlover as cl

import re
import os
import json
import datetime
from itertools import cycle

import math
import scipy.stats as stats
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize, scale

import sys
sys.path.append("../src/")
import auto_co2 as co2

from auto_co2.styles import generate_styles



########## Plotly Tools ##########

def save_plotly_fig(fig, format='png'):
    """
    Sauvegarde la figure figée au format PNG (Défaut) 
    ou la figue interactive (HTML)  
    ou la figure au fomat JSON
    dans le dossier output/figures
    """
    valid_formats = ['html', 'png', 'json']
    if format not in valid_formats:
        raise ValueError(f"Invalid format. Must be one of {valid_formats}")

    os.makedirs("../output/figures", exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = fig.layout.title.text if fig.layout.title.text else 'untitled'
    filename = f"{base_filename}_{timestamp}"
    filename = re.sub('[^a-zA-Z0-9-_]', '_', filename)

    if format == 'html':
        fig.write_html(f"../output/figures/{filename}.html")
    elif format == 'png':
        fig.write_image(f"../output/figures/{filename}.png")
    elif format == 'json':
        fig.write_json(f"../output/figures/{filename}.json")  



def load_plotly_json(filename):
    """
    Charghe la figure Plotly depuis un JSON
    Retourne un plotly.graph_objs._figure.Figure
    """

    output_dir = "../output/figures/"

    with open(f"{output_dir}{filename}", "r") as f:
        fig_dict = json.load(f)

    return go.Figure(fig_dict)


def add_legend(fig, text="Data Source: European Environment Agency, 2021"):
    fig.update_layout(
        annotations=[
            dict(
                x=1,  # Rightmost position
                y=-0.15,  # Keep it slightly below the plot
                showarrow=False,
                text=text,
                xref="paper",
                yref="paper",
                xanchor='right',  # Anchor the text to the right
                yanchor='auto'
            )
        ]
    )
    return fig

def increase_font_size(fig, font_size=24):
    fig.update_layout(
        height=400, 
        width=1200, 
        title_font=dict(size=font_size)
    )
    return fig


########## End of Plotly Tools ##########

########## Countrywise visualization #########
def co2_emissions_viz(countrie, save=True, format='png'): # Plotly Express
    per_co2 = countries.data.sort_values(by='Co2EmissionsWltp', ascending=False)
    fig = px.bar(per_co2, x='Country', y='Co2EmissionsWltp', color='Co2EmissionsWltp', color_continuous_scale='Blues')
    fig.update_layout(title={'text': "Quel pays achète les véhicules les plus polluants? (CO2)",'x': 0.3})
    fig.show()
    if save:
        save_plotly_fig(fig, format)
        

def engine_power_viz(countries, save=True, format='png'): # Plotly Express
    per_engine = countries.data.sort_values(by='EnginePower', ascending=False)
    fig = px.bar(per_engine, x='Country', y='EnginePower', color='EnginePower', color_continuous_scale='Greens')
    fig.update_layout(title={'text': "Quel pays achète les véhicules les plus puissants? (KWh)",'x': 0.3})
    fig.show()
    if save:
        save_plotly_fig(fig, format)
        

def mass_viz(countries, save=True, format='png'): # Plotly Express
    per_mass = countries.data.sort_values(by='MassRunningOrder', ascending=False)
    fig = px.bar(per_mass, x='Country', y='MassRunningOrder', color='MassRunningOrder', color_continuous_scale='Reds')
    fig.update_layout(title={'text': "Quel pays achète les véhicules les plus lourds? (Kg)",'x': 0.3})
    fig.show()
    if save:
        save_plotly_fig(fig, format)
        

def countrywise_viz(countries, save=True, format='png'): # Plotly Express
        co2_emissions_viz(countries, save=save, format=format)
        engine_power_viz(countries, save=save, format=format)
        mass_viz(countries, save=save, format=format)


########## End of countrywise visualization #########


########## Manufacturerwise visualization #########
def plot_popular_fueltype(manufacturers, save=True, format='png'):
    # Group by Pool and Make and sum the Counts for each group
    grouped_df = manufacturers.data.groupby(['Pool', 'Make'])['Count'].sum().reset_index(name='Counts')

    # Create a list of traces
    traces = []
    colors = cl.scales['12']['qual']['Paired']  # Get a list of 12 distinct colors
    make_colors = {make: colors[i % len(colors)] for i, make in enumerate(grouped_df['Make'].unique())}  # Create a color map
    
    for make in grouped_df['Make'].unique():
        df = grouped_df[grouped_df['Make'] == make]
        traces.append(go.Bar(x=df['Pool'], y=df['Counts'], name=make, marker_color=make_colors[make]))

    layout = go.Layout(barmode='stack', title="Répartition des types de carburants par marque/groupe automobile")
    fig = go.Figure(data=traces, layout=layout)
    fig.show()

    if save:
        save_plotly_fig(fig, format)





########## End of manufacturerwise visualization #########




######### Context visualization #########

def plot_registrations_per_month(df:pd.DataFrame, save=True, format='png'):
    mois = ['Janvier', 'Février', 'Mars', 'Avril', 
                        'Mai', 'Juin', 'Juil.', 'Août', 
                        'Sept.', 'Oct.', 'Nov.', 'Déc.']


    df['RegistrationDate'] = pd.to_datetime(df['RegistrationDate'])
    df['Month'] = df['RegistrationDate'].dt.month
    # Convert the groupby object to a DataFrame
    monthly_counts = df.groupby('Month').size().reset_index(name='counts')


    # Create a bar chart
    fig = px.bar(monthly_counts, x='Month', y='counts', color='counts', color_continuous_scale='Purples')

    # Update the layout
    fig.update_layout(
        title={
            'text': "Nombre d'immatriculations par mois",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="Mois",
        yaxis_title="Nombre d'immatriculations",
        showlegend=False
    )
    fig.update_xaxes(ticktext=mois, tickvals=list(range(1, 13)))
    if save:
        save_plotly_fig(fig, format)

    fig.show()

########## End context visualization ##########






######### Case spectific visualization #########

def plot_fueltype_distribution(df:pd.DataFrame, interactive=True, save=True, format='png'):
    """Affiche les boites à moustaches des émissions de CO2 par FuelType

    Args:
        df (DataFrame): _description_
        interactive (bool, optional): True par défaut, False pour de meilleures performances.

    Returns:
        _type_: plolty multi-boxplot figure
    """

    fig = go.Figure()
    for fuel_type in df['FuelType'].unique():
        fig.add_trace(go.Box(
            x=df[df['FuelType'] == fuel_type]['Co2EmissionsWltp'],
            name=fuel_type,
            boxpoints='outliers'))
        
    fig.update_layout(
        yaxis=dict(
            title='FuelType',
            tickangle=45),
        
        xaxis=dict(
            title='Co2EmissionsWltp'),
        
        autosize=False,
        width=1400,
        height=800,
        hovermode='x',  # Limit the number of hover labels
        annotations=[
            dict(
                x=0,
                y=0,
                showarrow=False,
                text="Data Source: European Environment Agency, 2021",
                xref="paper",
                yref="paper",
                font=dict(size=12))
        ]
    )
    
    if interactive:
        fig.show()
    else:
        img_bytes = pio.to_image(fig, format='png')
        display(Image(img_bytes))
        
    if save:
        save_plotly_fig(fig, format)





def plot_correlation_heatmap(df:pd.DataFrame, title='', interactive=True, save=True, format='png'):
    """Heatmap de corrélation des variables quantitatives

    Args:
        df (pd.DataFrame): Colonnes numériques uniquement
        interactive (bool, optional): True par défaut, False pour de meilleures performances.

    Returns:
        _type_: plotly heatmap figure
    """

    df = df.select_dtypes(include=[np.number])
    correlation_matrix = df.corr()
    heatmap = go.Heatmap(z=correlation_matrix, 
                         x=correlation_matrix.columns, 
                         y=correlation_matrix.columns, 
                         colorscale='RdBu_r')

    # Affichage des coefficients de corrélation
    annotations = []
    for i, row in enumerate(correlation_matrix.values):
        for j, value in enumerate(row):
            color  = 'white' if value < -0.5 or value > 0.5 else 'black'
            annotations.append(
                go.layout.Annotation(text=str(round(value, 2)), 
                                                    x=correlation_matrix.columns[j], 
                                                    y=correlation_matrix.columns[i], 
                                                    showarrow=False, 
                                                    font=dict(color=color, size=16)))

    fig = go.Figure(data=heatmap)
    fig.update_layout(
        title=dict(
            text=f'Corrélation des variables quantitatives {title}',
            x=0.55,  # Center the title
            font=dict(size=24)),
        autosize=False,
        width=1000,
        height=850,
        annotations=annotations,
        xaxis=dict(autorange='reversed', tickfont=dict(size=14)),
        yaxis=dict(tickfont=dict(size=14)))

    if interactive:
        fig.show()
    else:
        img_bytes = pio.to_image(fig, format='png')
        display(Image(img_bytes))
        
    if save:
        save_plotly_fig(fig, format)



def plot_qqplots(df:pd.DataFrame, title='', interactive=True, save=True, format='png'):
    n_samples = min(10000, df.shape[0])  # Sample 10,000 or the total number of rows, whichever is smaller
    df_sample = df.select_dtypes(include=[np.number]).sample(n=n_samples, random_state=1)

    # Create a subplot with 2 rows and 2 columns
    fig = make_subplots(rows=4, cols=2)

    # Loop over the first 4 columns of the DataFrame
    for i, col in enumerate(df_sample.columns[0:8]):
        # Calculate the theoretical quantiles and order them
        theoretical_quantiles = np.sort(stats.norm.ppf((np.arange(len(df_sample[col])) + 0.5) / len(df_sample[col])))
        
        # Calculate the sample quantiles and order them
        sample_quantiles = np.sort(df_sample[col])
        
        fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sample_quantiles, mode='markers', name=col), row=(i//2)+1, col=(i%2)+1)

    # Update layout
    fig.update_layout(height=1000, width=800, title_text=f"Quantiles observés vs quantiles d'une distribution normale (QQ Plots) {title}")

    if interactive:
        fig.show()
    else:
        img_bytes = pio.to_image(fig, format='png')
        display(Image(img_bytes))
        
    if save:
        save_plotly_fig(fig, format)



def plot_feature_distributions(df, title='', interactive=True, save=True, format='png'):
    fig = sp.make_subplots(rows=3, cols=3)
    df_num = df.select_dtypes(include=[np.number])
    cols = [col for col in df_num.columns]  # Only keep numeric columns

    for i, col_name in enumerate(cols):
        row = i // 3 + 1
        col = i % 3 + 1
        fig.add_trace(go.Histogram(x=df[col_name], nbinsx=40, name=col_name), row=row, col=col)
        fig.update_xaxes(title_text=col_name, row=row, col=col)  # Add x-axis label

    fig.update_layout(height=1000, width=1100, title_text=f"Distribution des variables explicatives {title}", showlegend=False)  # Add top-level title and remove legend

    if interactive:
        fig.show()
    else:
        img_bytes = pio.to_image(fig, format='png')
        display(Image(img_bytes))
        
    if save:
        save_plotly_fig(fig, format)
        
        
def plot_distribution_pie(target, title='', interactive=True, save=True, format='png'):
    counts = pd.Series(target).value_counts()

    fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values, hole=.3, textinfo="label+percent")])

    fig.update_layout(
        autosize=False,
        margin=dict(t=50, b=50, l=50, r=50),
        title_text=f"Répartition des classes {title}",
        title_x=0.5
    )

    if interactive:
        fig.show()
    else:
        img_bytes = pio.to_image(fig, format=format)
        display(Image(img_bytes))
        
    if save:
        save_plotly_fig(fig, format)
        
######### End of case specific visualization #########


########## Model Visualizations ##########


def plot_confusion_matrix(y_true, y_pred, 
                          palette='Blues', 
                          classes=None, 
                          interactive=True, 
                          save=True,
                          format='png', 
                          title=''):
    
    cm = confusion_matrix(y_true, y_pred)
    if classes is None:
        classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    else:
        classes = list(range(cm.shape[0]))

    heatmap = go.Heatmap(z=cm, 
                         x=classes, 
                         y=classes, 
                         colorscale=palette)

    annotations = []
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            color  = 'white' if value > cm.max() / 2 else 'black'
            annotations.append(
                go.layout.Annotation(text=str(value), 
                                     x=j, 
                                     y=i, 
                                     showarrow=False, 
                                     font=dict(color=color, size=16)))

    fig = go.Figure(data=heatmap)
    fig.update_layout(
        title=dict(
            text=f"Matrice de confusion {title}",
            x=0.5,  # Center the title
            font=dict(size=24)),
        autosize=False,
        width=600,
        height=600,
        annotations=annotations,
        xaxis=dict(title='Labels prédits', tickfont=dict(size=14)),
        yaxis=dict(title='Vrais labels', tickfont=dict(size=14)))

    if interactive:
        fig.show()
    else:
        img_bytes = pio.to_image(fig, format=format)
        display(Image(img_bytes))
        
    if save:
        save_plotly_fig(fig, format)
        


def plot_regression_diagnostics(y_test, pred_test, y_train, title, interactive=True, save=True, format='png'):
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Valeurs réelles VS valeurs prédites (test)",
                                                       "Répartition des résidus (test)",
                                                       "Distribution des résidus (test)",
                                                       "Diagramme Quantile-Quantile"))

    # Actual vs Predicted values
    residuals = pred_test - y_test
    fig.add_trace(go.Scatter(x=y_test, y=pred_test, mode='markers', marker=dict(color='#2E8B57', size=5, opacity=0.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], mode='lines', line=dict(color='red')), row=1, col=1)

    # Residuals Distribution
    fig.add_trace(go.Scatter(x=y_test, y=residuals, mode='markers', marker=dict(color='#980a10', size=5, opacity=0.1)), row=1, col=2)
    fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[0, 0], mode='lines', line=dict(color='#0a5798')), row=1, col=2)

    # Histogramme des résidus
    fig.add_trace(go.Histogram(x=residuals, nbinsx=40), row=2, col=1)

    # Q-Q plot
    residuals_norm = scale(residuals)
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals_norm, dist='norm', fit=True)
    fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers'), row=2, col=2)
    fig.add_trace(go.Scatter(x=osm, y=slope*osm + intercept, mode='lines'), row=2, col=2)

    fig.update_layout(height=1000, width=1300, showlegend=False, title_text=title, title_x=0.5, title_font=dict(size=24))
    
    if interactive:
        fig.show()
    else:
        img_bytes = pio.to_image(fig, format=format)
        display(Image(img_bytes))
        
    if save:
        save_plotly_fig(fig, format)
        
        
def plot_lr_elasticnet(lr_en, title='', interactive=True, save=True, format='png'):
    alphas = lr_en.alphas_

    fig = go.Figure()
    for i in range(lr_en.mse_path_.shape[0]):
        fig.add_trace(go.Scatter(x=alphas, y=lr_en.mse_path_[i,:,:].mean(axis=1), mode='lines', name='Moyenne pour l1_ratio= %.2f' %lr_en.l1_ratio[i]))

    fig.update_layout(title='Mean squared error pour chaque lambda'+title, xaxis_title='Alpha', yaxis_title='Mean squared error')

    if interactive:
        fig.show()
    else:
        img_bytes = pio.to_image(fig, format=format)
        display(Image(img_bytes))
        
    if save:
        save_plotly_fig(fig, format)
        
        
def plot_mse_folds(lr_en, l1_ratios):
    # Define a color for each fold number
    colors = ['red', 'green', 'blue', 'orange', 'purple']

    # Calculate the number of rows and columns for the subplots
    n_rows = math.ceil(len(l1_ratios) / 2)
    n_cols = 2 if len(l1_ratios) > 1 else 1

    # Create subplots
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[f'l1_ratio={l1_ratio}' for l1_ratio in l1_ratios])

    # Add a line for each fold in each subplot
    for i, l1_ratio in enumerate(l1_ratios):
        row = i // n_cols + 1
        col = i % n_cols + 1
        for j, fold in enumerate(lr_en.mse_path_[i]):
            fig.add_trace(go.Scatter(x=lr_en.alphas_, y=fold, mode='lines', name=f'Fold {j+1}', line=dict(color=colors[j])), row=row, col=col)

        # Add a line for the average MSE across folds in each subplot
        avg_mse = lr_en.mse_path_[i].mean(axis=0)
        fig.add_trace(go.Scatter(x=lr_en.alphas_, y=avg_mse, mode='lines', name='Average across the folds', line=dict(color='black', width=2, dash='dot')), row=row, col=col)

        # Add a vertical line for the chosen alpha in each subplot
        fig.add_shape(type='line', x0=lr_en.alpha_, x1=lr_en.alpha_, y0=avg_mse.min(), y1=avg_mse.max(), line=dict(color='black', dash='dash'), row=row, col=col)

    # Update y-axes to have the same scale
    fig.update_yaxes(matches='y')
    fig.update_layout(height=300*n_rows, width=600*n_cols, title_text="Mean square error on each fold")
    fig.update_xaxes(title_text='Alphas')
    fig.show()
                


def plot_pca_variance(pca, n_features,  title='', interactive=True, save=True, format='png'):

    x_values = list(range(1, n_features+1))

    # Create a scatter plot of the explained variance ratio
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=pca.explained_variance_ratio_[:n_features], mode='lines+markers', name='Variance'))
    fig.add_trace(go.Scatter(x=x_values, y=np.cumsum(pca.explained_variance_ratio_)[:n_features], mode='lines+markers', name='Cumulative Variance'))

    fig.update_layout(title='Part de la variance expliquée, composants PCA'+title,
                      xaxis=dict(title='Component'),
                      yaxis=dict(title='Part de la variance expliquée'))
    fig.update_layout(height=700, width=1000)


    if interactive:
        fig.show()
    else:
        img_bytes = pio.to_image(fig, format=format)
        display(Image(img_bytes))
        
    if save:
        save_plotly_fig(fig, format)
        
        
        
def plot_xgboost(results, metric='mlogloss', title='', interactive=True, save=True, format='png'):
    num_epochs = len(next(iter(results.values()))[metric])
    x_axis = list(range(0, num_epochs))

    fig = go.Figure()
    for key in results.keys():
        fig.add_trace(go.Scatter(x=x_axis, y=results[key][metric], mode='lines', name=key))

    fig.update_layout(title='XGBoost'+title, xaxis_title='Epoch', yaxis_title=metric)
    if interactive:
        fig.show()
    else:
        img_bytes = pio.to_image(fig, format=format)
        display(Image(img_bytes))
        
    if save:
        save_plotly_fig(fig, format)    
    

def plot_roc_curves(y_test, y_pred, title='', interactive=True, save=True, format='png'):
    # Determine unique classes
    classes = np.unique(y_test)

    # Binarize the output
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = y_test_bin.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Create a Figure
    fig = go.Figure()

    # Add ROC curves to the Figure
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'blue', 'yellow'])
    for i, color in zip(range(n_classes), colors):
        fig.add_trace(go.Scatter(x=fpr[i], y=tpr[i], mode='lines', name='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]), line=dict(color=color)))

    # Add diagonal line
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='black', dash='dash')))

    # Update layout
    fig.update_layout(title='Courbes ROC: '+title, xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', autosize=False, width=600, height=600, margin=dict(l=50, r=50, b=100, t=100, pad=4))
    fig.update_layout(height=700, width=1000)

    # Show or save the Figure
    if interactive:
        fig.show()
    else:
        img_bytes = pio.to_image(fig, format=format)
        display(Image(img_bytes))
        
    if save:
        save_plotly_fig(fig, format)
        
        

def plot_feature_importance(model, max_num_features=20, title='', interactive=True, save=True, format='png'):
    
    importance = model.get_booster().get_score(importance_type='weight')
    sorted_importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=False))
    limited_importance = dict(list(sorted_importance.items())[:max_num_features])

    fig = go.Figure()
    for key in limited_importance.keys():
        fig.add_trace(go.Bar(y=[key], x=[limited_importance[key]], orientation='h', name=key))

    fig.update_layout(title='Feature Importance'+title, yaxis_title='Features', xaxis_title='Importance')
    fig.update_layout(height=700, width=1000)

    if interactive:
        fig.show()
    else:
        img_bytes = pio.to_image(fig, format=format)
        display(Image(img_bytes))
        
    if save:
        save_plotly_fig(fig, format)
        
        
        
def plot_training_history(training_history, title='', interactive=True, save=True, format='png'):
    fig = make_subplots(rows=2, cols=1)

    fig.add_trace(go.Scatter(y=training_history.history['accuracy'], mode='lines', name='Train Accuracy'), row=1, col=1)
    fig.add_trace(go.Scatter(y=training_history.history['val_accuracy'], mode='lines', name='Validation Accuracy'), row=1, col=1)
    fig.add_trace(go.Scatter(y=training_history.history['loss'], mode='lines', name='Train Loss'), row=2, col=1)
    fig.add_trace(go.Scatter(y=training_history.history['val_loss'], mode='lines', name='Validation Loss'), row=2, col=1)

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)

    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=2, col=1)

    fig.update_layout(height=600, width=600, title_text="Model Accuracy and Loss" + title)

    if interactive:
        fig.show()
    else:
        img_bytes = pio.to_image(fig, format=format)
        display(Image(img_bytes))
        
    if save:
        save_plotly_fig(fig, format)
        
        

def plot_shap_summary(shap_explanation, feature_names, max_num_features=20, title='', interactive=True, save=True, format='png'):
    # Get SHAP values from the explanation object
    shap_values = shap_explanation.values

    # Calculate the mean absolute SHAP values for each feature
    mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)

    # Create a dictionary of feature names and mean absolute SHAP values
    importance = dict(zip(feature_names, mean_abs_shap_values))

    # Sort features according to importance
    sorted_importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=False))

    # Limit the number of features
    limited_importance = dict(list(sorted_importance.items())[:max_num_features])

    # Create a bar chart
    fig = go.Figure()
    for key in limited_importance.keys():
        fig.add_trace(go.Bar(y=[key], x=[limited_importance[key]], orientation='h', name=key))

    fig.update_layout(title='SHAP Summary'+title, yaxis_title='Features', xaxis_title='Importance')
    if interactive:
        fig.show()
    else:
        img_bytes = pio.to_image(fig, format=format)
        display(Image(img_bytes))
        
    if save:
        save_plotly_fig(fig, format)