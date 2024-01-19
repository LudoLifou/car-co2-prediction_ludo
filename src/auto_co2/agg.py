import numpy as np
import pandas as pd
import requests
import io
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import colorlover as cl

from auto_co2.styles import *
from auto_co2.viz import *



########## Classes ##########
class CountryDataAggregator:
    """_summary_
    """
    def __init__(self, df:pd.DataFrame, country_stats: dict=None):
        if country_stats is None:
            country_stats = country_stats_query_230110
        required_columns = ['Country', 'Make', 'FuelType', 'FuelConsumption', 
                            'Co2EmissionsWltp', 'MassRunningOrder', 'BaseWheel', 'AxleWidthSteering']
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"Le DataFrame doit contenir les colonnes: {required_columns}")
        
        pd.set_option('display.float_format', '{:.3f}'.format)       
        
        self.data = df.groupby('Country').agg({
            'ID': 'count',
            'Make': lambda x: x.mode()[0],
            'FuelType': lambda x: x.mode()[0],
            'FuelConsumption': 'mean',
            'EnginePower': 'mean',
            'Co2EmissionsWltp': 'mean',
            'MassRunningOrder': 'mean',
            'BaseWheel': 'mean',
            'AxleWidthSteering': 'mean',
            'ElectricRange': 'mean'})
        
        
        self.data = self.data.rename(columns={'ID': 'Count'})
        
        country_stats = pd.DataFrame(country_stats).transpose()
        country_stats.index.name = 'Country'
        self.data = self.data.join(country_stats, how='inner')

        gdp = self.data.pop('gdp_per_capita')
        self.data.insert(0, 'GdpPerCapita', gdp)
        
        pop = self.data.pop('population')        
        self.data.insert(0, 'Population', pop)

        # Convert the index to a column
        self.data.reset_index(inplace=True)
                
    def __repr__(self):
        return f"Countries(data={self.data.head()})"
    
    def sort(self, by='population', ascending=False):
        self.data.sort_values(by=by, ascending=ascending, inplace=True)

    def display(self, columns=None, n=None, styles=None, title=None):
        if styles is None:
            styles = generate_styles()
        if columns is not None:
            print(f"Columns: {columns}")  # Debugging line
            print(f"DataFrame columns: {self.data.columns}")  # Debugging line
            data = self.data[columns]
        else:
            data=self.data
            
        displayer(data, n, styles, title)
        
    def display_sorted(self, by='population', ascending=False, columns=None, n=None, styles=None, title=None):
        self.sort(by=by, ascending=ascending)
        self.display(columns, n, styles, title)
        
    def co2_emissions_viz(self, save=True, format='png'): # Plotly Express
        per_co2 = self.data.sort_values(by='Co2EmissionsWltp', ascending=False)
        fig = px.bar(per_co2, x='Country', y='Co2EmissionsWltp', color='Co2EmissionsWltp', color_continuous_scale='Blues')
        fig.update_layout(title={'text': "Quel pays achète les véhicules les plus polluants? (CO2)",'x': 0.3})
        fig = add_legend(fig)
        fig.show()
        if save:
            save_plotly_fig(fig, format)
            

    def engine_power_viz(self, save=True, format='png'): # Plotly Express
        per_engine = self.data.sort_values(by='EnginePower', ascending=False)
        fig = px.bar(per_engine, x='Country', y='EnginePower', color='EnginePower', color_continuous_scale='Greens')
        fig.update_layout(title={'text': "Quel pays achète les véhicules les plus puissants? (KWh)",'x': 0.3})
        fig = add_legend(fig)
        fig.show()
        if save:
            save_plotly_fig(fig, format)
            

    def mass_viz(self, save=True, format='png'): # Plotly Express
        per_mass = self.data.sort_values(by='MassRunningOrder', ascending=False)
        fig = px.bar(per_mass, x='Country', y='MassRunningOrder', color='MassRunningOrder', color_continuous_scale='Reds')
        fig.update_layout(title={'text': "Quel pays achète les véhicules les plus lourds? (Kg)",'x': 0.3})
        fig = add_legend(fig)        
        fig.show()
        if save:
            save_plotly_fig(fig, format)
            

    def countrywise_viz(self, save=True, format='png'): # Plotly Express
            self.co2_emissions_viz(save=save, format=format)
            self.engine_power_viz(save=save, format=format)
            self.mass_viz(save=save, format=format)




class ManufacturerDataAggregator:
    def __init__(self, df:pd.DataFrame):
        required_columns = [
            'Make', 'FuelType', 'FuelConsumption', 
            'Co2EmissionsWltp', 'MassRunningOrder', 'BaseWheel', 'AxleWidthSteering']

        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"Le DataFrame doit contenir les colonnes: {required_columns}")

        pd.set_option('display.float_format', '{:.3f}'.format)

        self.data = df.groupby(['Pool', 'Make']).agg({
            'Make': 'count',
            'FuelType': lambda x: x.mode()[0],
            'FuelConsumption': 'mean',
            'EnginePower': 'mean',
            'Co2EmissionsWltp': 'mean',
            'MassRunningOrder': 'mean',
            'BaseWheel': 'mean',
            'AxleWidthSteering': 'mean',
            'ElectricRange': 'mean'})
        
        self.data = self.data.rename(columns={'Make': 'Count'})
        
        total_count = self.data['Count'].sum()
        market_share = ((self.data['Count'] / total_count) * 100).round(2)
        self.data.insert(1, 'MarketShare(%)', market_share)
        # Filter out rows where 'Count' is less than 0.2% of total_count
        self.data = self.data[self.data['Count'] >= 0.002 * total_count]
                
    def __repr__(self):
        return f"Manufacturers(data={self.data.head()})"
    
    def sort(self, by='Count', ascending=False):
        self.data.sort_values(by=by, ascending=ascending, inplace=True)

    def display(self, n=None, styles=None, title=None):
        if styles is None:
            styles = generate_styles()
        displayer(self.data, n, styles, title)
    
    def display_sorted(self, by='Count', ascending=False, n=None, styles=None, title=None):
        self.sort(by=by, ascending=ascending)
        self.display(n, styles, title)
        
    def plot_popular_fueltype(self, save=True, format='png'):
        # Group by Pool and Make and sum the Counts for each group
        grouped_df = self.data.groupby(['Pool', 'Make'])['Count'].sum().reset_index(name='Counts')

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
            
    def plot_mass_engine_scatter(self):
        unique_df = self.data.reset_index(level='Pool').drop_duplicates(subset=['Count'])

        # Create a scatter plot
        fig = px.scatter(
            unique_df,
            x="MassRunningOrder",
            y="EnginePower",
            color="Pool",
            hover_name="Count", 
            labels={
                "MassRunningOrder": "Masse du véhicule",
                "EnginePower": "Puissance du moteur",
            },
            title="Comparaison de la masse et de la puissance des véhicules (KW)",
        )
        fig = add_legend(fig)
        fig.show()
        


class CarDataAggregator:
    def __init__(self, df:pd.DataFrame):
        required_columns = [
            'Make', 'FuelConsumption', 
            'Co2EmissionsWltp', 'MassRunningOrder', 'BaseWheel', 'AxleWidthSteering']

        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"Le DataFrame doit contenir les colonnes: {required_columns}")

        pd.set_option('display.float_format', '{:.3f}'.format)

        # Handle NaN values
        for column in ['FuelConsumption', 'EnginePower', 'Co2EmissionsWltp', 'MassRunningOrder', 'BaseWheel', 'AxleWidthSteering', 'ElectricRange']:
            df[column] = df[column].fillna(df[column].mean())
        df = df.dropna(subset=['Pool'])
        # Add 'CarCount' column
        df['CarCount'] = df.groupby('CommercialName')['CommercialName'].transform('count')

        self.data = df.groupby(['Make', 'CommercialName']).agg({
            'Pool': lambda x: x.mode()[0],
            'FuelConsumption': 'mean',
            'EnginePower': 'mean',
            'Co2EmissionsWltp': 'mean',
            'MassRunningOrder': 'mean',
            'BaseWheel': 'mean',
            'AxleWidthSteering': 'mean',
            'ElectricRange': 'mean',
            'CarCount': 'first'  # Take the first 'CarCount' value for each group
        })     

        features = ['MassRunningOrder', 'EnginePower', 'BaseWheel', 'Co2EmissionsWltp', 'ElectricRange']  
        self.min_values = self.data[features].min()
        self.max_values = self.data[features].max()
                
    def __repr__(self):
        return f"Cars(data={self.data.head()})"
    
    def display(self, data=None, n=None, styles=None, title=None):
        if data is None:
            data = self.data.loc[:, self.data.columns != 'CarCount']
        if styles is None:
            styles = generate_styles()

        displayer(data, n, styles, title)

    def sort(self, by='Co2EmissionsWltp', ascending=False):
        df = self.data.copy()
        df.sort_values(by=by, ascending=ascending, inplace=True)
        return df
    
    def display_sorted(self, by='Count', ascending=False, n=None, styles=None, title=None):
        df = self.sort(by=by, ascending=ascending)
        self.display(df, n, styles, title)
    
    def specs(self, car_names, styles=None):
        if styles is None:
            styles = generate_styles()

        # Ensure that self.data is a MultiIndex DataFrame
        if not isinstance(self.data.index, pd.MultiIndex):
            self.data.set_index(['Make', 'CommercialName'], inplace=True)

        selected_cars = [self.data.xs(car_name, level='CommercialName').T.rename(columns=lambda x: car_name) for car_name in car_names]
        selected_cars_df = pd.concat(selected_cars, axis=1)
        displayer(selected_cars_df, styles=styles, n=None)
            

    def plot_polar_charts(self, selected_vehicles, title):
        features = ['MassRunningOrder', 'EnginePower', 'BaseWheel', 'Co2EmissionsWltp', 'ElectricRange']  
        min_values = self.data[features].min()  # Get min values from the entire dataset
        max_values = self.data[features].max()  # Get max values from the entire dataset

        subplot_titles = selected_vehicles.index.get_level_values('CommercialName').tolist()
        fig = make_subplots(rows=1, cols=3, subplot_titles=subplot_titles, specs=[[{'type': 'polar'}]*3])
        
        for i, (idx, row) in enumerate(selected_vehicles.iterrows()):   # Add a polar chart for each 'CommercialName'
            normalized_data = [(row[feature] - min_values[feature]) / (max_values[feature] - min_values[feature]) * 100 for feature in features]
            car_name = ' '.join(idx)  # Join 'Make' and 'CommercialName' to form the car name
            fig.add_trace(go.Scatterpolar(
                r=normalized_data,
                theta=features,
                fill='toself',
                name=car_name
            ), row=1, col=i+1)

        fig.update_layout(title_text=title)
        fig = increase_font_size(fig, font_size=30)        
        for i in range(1, 4):
            fig.update_polars(radialaxis_range=[0, 100], radialaxis_rotation=20, selector={"polar": f"polar{i}"})
        
        fig = add_legend(fig)
        fig.show()

    def plot_top_vehicles(self):
        # Select 3 vehicles based on 'CarCount', 'Co2EmissionsWltp' and least 'Co2EmissionsWltp'
        top_3_count = self.data.nlargest(3, 'CarCount')
        top_3_co2 = self.data.nlargest(3, 'Co2EmissionsWltp')
        bottom_3_co2 = self.data.nsmallest(3, 'Co2EmissionsWltp')

        # Plot polar charts for each selection
        self.plot_polar_charts(top_3_count, "Meilleures ventes 2021")
        self.plot_polar_charts(top_3_co2, "Voitures les plus polluantes")
        self.plot_polar_charts(bottom_3_co2, "Voitures les moins polluantes")
        
########### End of classes ##########




########## Toolkit ##########

def float_rounder(df, n=3):
        float_cols = df.select_dtypes(include='float64').columns
        return df[float_cols].round(n)

def rename_dict_keys(d:dict, old_keys:list, new_keys:list):
    if len(old_keys) != len(new_keys):
        raise ValueError("Both key lists must have the same length")
    for old_key, new_key in zip(old_keys, new_keys):
        d[new_key] = d.pop(old_key)
        
        
    

########## End of toolkit ##########





########## External sources ##########
def get_country_population(): # WIKIPEDIA QUERY
    url = 'https://query.wikidata.org/sparql'

    query = """
    SELECT ?countryCode ?population
    WHERE
    {
        ?country wdt:P463 wd:Q458; wdt:P297 ?countryCode; wdt:P1082 ?population.
    }
    """

    data = requests.get(url, params={'format': 'json', 'query': query}).json()
    country_population = {item['countryCode']['value']: int(item['population']['value']) for item in data['results']['bindings']}
    return country_population #dict

def get_country_gdp(): # WIKIPEDIA QUERY
    url = 'https://query.wikidata.org/sparql'

    query = """
    SELECT ?countryCode ?gdp
    WHERE
    {
        ?country wdt:P463 wd:Q458; wdt:P297 ?countryCode; wdt:P2131 ?gdp.
    }
    """

    data = requests.get(url, params={'format': 'json', 'query': query}).json()
    country_gdp = {item['countryCode']['value']: float(item['gdp']['value']) for item in data['results']['bindings']}
    return country_gdp #dict

def get_country_stats(): # WIKIPEDIA QUERY
    country_population = get_country_population()
    country_gdp = get_country_gdp()

    country_stats = {}
    for countryCode in country_population.keys():
        if countryCode in country_gdp:
            country_stats[countryCode] = {
                'population': country_population[countryCode],
                'gdp_per_capita': country_gdp[countryCode] / country_population[countryCode]
            }

    return country_stats

# Erreur de virgule sur les données de la Hongrie, les stats seront passées en variable:
country_stats_query_230110 = {
 'IE': {'population': 5123536, 'gdp_per_capita': 103296.79936336935},
 'HU': {'population': 9603634, 'gdp_per_capita': 18935.334502543515},
 'ES': {'population': 47415750, 'gdp_per_capita': 30103.513733200467},
 'BE': {'population': 11584008, 'gdp_per_capita': 51785.18523122567},
 'LU': {'population': 660809, 'gdp_per_capita': 129396.30639715864},
 'FI': {'population': 5563970, 'gdp_per_capita': 53269.09118201572},
 'SE': {'population': 10548336, 'gdp_per_capita': 60375.042698298574},
 'DK': {'population': 5827463, 'gdp_per_capita': 68349.34391930074},
 'PL': {'population': 38382576, 'gdp_per_capita': 17929.40124589345},
 'LT': {'population': 2860002, 'gdp_per_capita': 24592.39504308039},
 'IT': {'population': 58850717, 'gdp_per_capita': 35927.442598770715},
 'AT': {'population': 8979894, 'gdp_per_capita': 53493.772186286384},
 'GR': {'population': 10482487, 'gdp_per_capita': 20898.272753975274},
 'PT': {'population': 10347892, 'gdp_per_capita': 24544.40455804912},
 'FR': {'population': 67749632, 'gdp_per_capita': 43658.97897812936},
 'GB': {'population': 67326569, 'gdp_per_capita': 46378.1296493068},
 'DE': {'population': 83149300, 'gdp_per_capita': 51232.36048676297},
 'EE': {'population': 1373101, 'gdp_per_capita': 27748.00466899376},
 'LV': {'population': 1883008, 'gdp_per_capita': 21855.4104193928},
 'CZ': {'population': 10827529, 'gdp_per_capita': 23152.189202171612},
 'SK': {'population': 5449270, 'gdp_per_capita': 21774.768346769382},
 'SI': {'population': 2066880, 'gdp_per_capita': 30053.882187161325},
 'RO': {'population': 19053815, 'gdp_per_capita': 15811.089953586723},
 'BG': {'population': 7000039, 'gdp_per_capita': 12719.986046649168},
 'HR': {'population': 3871833, 'gdp_per_capita': 18328.426475263783},
 'CY': {'population': 1141166, 'gdp_per_capita': 24921.04806925548},
 'MT': {'population': 465292, 'gdp_per_capita': 38180.90578604403},
 'NL': {'population': 17100715, 'gdp_per_capita': 48569.46734975701}}

########## End External sources ##########