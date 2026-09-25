# App-Insights-Analysis- Google Play Store
## Project Overview
This project performs an end-to-end Exploratory Data Analysis (EDA) on the Google Play Store dataset to uncover key factors influencing app ratings, install volumes, pricing viability, and update cadences.

## Tech Stack & Libraries
- Python
- Pandas & NumPy (Data Cleaning, Transformation & Grouped Aggregations)
- Matplotlib & Seaborn (Statistical Visualizations)

## Key Insights & Findings
1. **Catalog Benchmarks:** The platform-wide average rating is ~4.19/5.0, with Free apps accounting for over 90% of total listings.
2. **Correlation Analysis:** Pearson correlation between Installs and Ratings is near-zero (~0.05), proving that scale and user satisfaction operate independently.
3. **Monetization & Pricing:** Apps priced between $1.00 and $5.00 maintain steady rating averages, whereas apps above $50.00 exhibit sharp rating declines due to heightened user expectations.
4. **Maintenance Impact:** More than 70% of active apps received updates in 2018, demonstrating a strong link between active maintenance and store visibility.
5. **Genre Benchmarks:** Specialized categories (e.g., Comics, Education) sustain higher median satisfaction scores than high-friction casual categories.

## Visualizations

### App Update Volume Trend (2016 - 2018)
![App Update Volume Trend](Images/Q2_%20App%20Update%20Volume%20Trend%20(2016%20-%202018).png)

### App Rating Distribution by Install Tier
![App Rating Distribution](Images/Q3_%20App%20Rating%20Distribution%20by%20Install%20Tier.png)

### Highest vs Lowest Rated Genres
![Highest vs Lowest Rated Genres](Images/Q5_%20Highest%20vs%20Lowest%20Rated%20Genres%20(Mean%20Rating,%20N%20_=%2030).png)

### Size vs Installs
![Size vs Installs](Images/Scatter%20Plot_%20Size%20vs%20Installs.png)

### Top 5 Categories by Total Installs
![Top 5 Categories](Images/Top%205%20Categories%20by%20Total%20Installs%20(Billions).png)

### App Size Distribution
![App Size Distribution](Images/app_size_distribution.png)

