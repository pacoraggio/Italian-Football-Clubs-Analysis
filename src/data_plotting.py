import pandas as pd  
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

def plot_country_metrics(df: pd.DataFrame, countries: list,  metric: str, figsize = (12, 6)):
    '''Plot the specified metric for the given countries and seasons.
    
    Each country has a unique line marker. Where data is NaN (indicating seasons 
    where no teams from that country participated), a special 'X' marker is displayed
    to indicate missing data.
    '''
    
    # Define different markers for each country
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Metric descriptions for meaningful titles
    metric_descriptions = {
        'total_matches': 'Total Matches per Season',
        'win_rate': 'Win Rate (%)',
        'loss_rate': 'Loss Rate (%)',
        'draw_rate': 'Draw Rate (%)',
        'ppg_3': 'Points Per Game (3pts for Win)',
        'ppg_2': 'Points Per Game (2pts for Win)',
        'goal_diff': 'Goal Difference',
        'gf_pg': 'Goals For per Game',
        'ga_pg': 'Goals Against per Game',
        'gdpg': 'Goal Difference per Game'
    }
    
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    
    dataset = df[df['country'].isin(countries)].copy()
    
    # Get color palette for consistent coloring
    palette = sns.color_palette("husl", len(countries))
    color_map = {country: palette[i] for i, country in enumerate(countries)}
    
    # Track if we've added the NaN label (add only once to legend)
    nan_label_added = False
    
    for i, country in enumerate(countries):
        country_data = dataset[dataset['country'] == country].sort_values('season')
        marker = markers[i % len(markers)]
        color = color_map[country]
        
        # Plot line with markers for non-NaN values (gaps appear at NaN)
        ax.plot(country_data['season'], country_data[metric], 
                marker=marker, label=country, linewidth=1.5, 
                markersize=4, color=color)
        
        # Add special 'X' markers for NaN values to indicate "no data"
        nan_mask = country_data[metric].isna()
        if nan_mask.any():
            nan_seasons = country_data.loc[nan_mask, 'season'].values
            # Position NaN markers slightly below the minimum y value
            y_min = ax.get_ylim()[0]
            nan_y = [y_min] * len(nan_seasons)
            
            ax.scatter(nan_seasons, nan_y, marker='x', s=200, 
                      color=color, linewidths=1.5, alpha=0.6, zorder=5,
                      label='No matches' if not nan_label_added else '')
            
            if not nan_label_added:
                nan_label_added = True
    
    # Adjust y-axis to accommodate NaN markers at the bottom
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min - (y_max - y_min) * 0.12, y_max)
    
    # Use descriptive title if metric is in the descriptions dictionary
    metric_title = metric_descriptions.get(metric, metric)
    
    plt.xticks(rotation=70)
    plt.title(f"{metric_title} by Country over Seasons")
    plt.xlabel("Season")
    plt.ylabel(metric_title)
    plt.legend(title='Country', fontsize=9)
    plt.tight_layout()
    plt.show()