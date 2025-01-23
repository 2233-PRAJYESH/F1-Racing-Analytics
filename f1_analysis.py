import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the data
df = pd.read_csv('lead_data.csv')

# Basic statistical analysis
def analyze_racing_data(df):
    # Display basic statistics
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Calculate correlation between speed and lap time
    correlation = df['average_speed (km/h)'].corr(df['lap_time (seconds)'])
    print(f"\nCorrelation between speed and lap time: {correlation:.2f}")
    
    # Team performance analysis
    team_stats = df.groupby('team').agg({
        'average_speed (km/h)': 'mean',
        'lap_time (seconds)': 'mean',
        'position_won': 'mean'
    }).round(2)
    print("\nTeam Performance Stats:")
    print(team_stats)
    
    # Visualizations
    plt.figure(figsize=(15, 10))
    
    # Speed vs Position plot
    plt.subplot(2, 2, 1)
    plt.scatter(df['average_speed (km/h)'], df['position_won'])
    plt.xlabel('Average Speed (km/h)')
    plt.ylabel('Position')
    plt.title('Speed vs Position')
    
    # Lap Time vs Position plot
    plt.subplot(2, 2, 2)
    plt.scatter(df['lap_time (seconds)'], df['position_won'])
    plt.xlabel('Lap Time (seconds)')
    plt.ylabel('Position')
    plt.title('Lap Time vs Position')
    
    # Team Performance (Average Speed)
    plt.subplot(2, 2, 3)
    sns.barplot(x=df['team'], y=df['average_speed (km/h)'])
    plt.xticks(rotation=45)
    plt.title('Average Speed by Team')
    
    # Team Performance (Average Position)
    plt.subplot(2, 2, 4)
    sns.barplot(x=df['team'], y=df['position_won'])
    plt.xticks(rotation=45)
    plt.title('Average Position by Team')
    
    plt.tight_layout()
    plt.show()
    
    # Find the fastest lap
    fastest_lap = df.loc[df['lap_time (seconds)'].idxmin()]
    print("\nFastest Lap Details:")
    print(f"Racer: {fastest_lap['racer_name']}")
    print(f"Team: {fastest_lap['team']}")
    print(f"Time: {fastest_lap['lap_time (seconds)']} seconds")
    print(f"Speed: {fastest_lap['average_speed (km/h)']} km/h")

if __name__ == "__main__":
    analyze_racing_data(df)
