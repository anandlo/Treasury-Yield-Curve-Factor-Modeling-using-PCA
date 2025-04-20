# Yield Curve Analysis using Principal Component Analysis (PCA)

## Overview

This project explores the dynamics of the US Treasury yield curve using dimensionality reduction techniques, specifically Principal Component Analysis (PCA). Understanding and modeling the yield curve is crucial for various financial applications, including portfolio management, trading, and risk management.

The yield curve typically shows interest rates rising with maturity due to the time value of money. Research has shown that changes in the yield curve's shape (shifts, twists) can often be attributed to a few underlying, unobservable factors. These are commonly identified as **level**, **slope**, and **curvature**.

This project aims to:
1.  Fetch historical US Treasury yield data for various maturities.
2.  Perform Exploratory Data Analysis (EDA) to understand the data characteristics.
3.  Apply PCA to reduce the dimensionality of the yield curve data.
4.  Identify and interpret the principal components corresponding to level, slope, and curvature movements.
5.  Demonstrate how the original yield curve data can be reconstructed using a reduced number of components.

This technique is valuable for traders and risk managers who need to condense yield curve movements into key risk factors for hedging interest rate risk effectively.

## Features

* Fetches US Treasury yield curve data using the Quandl API.
* Handles data cleaning (missing value imputation).
* Performs data standardization for PCA.
* Applies PCA to identify key drivers of yield curve movements.
* Visualizes the principal components and their interpretation (level, slope, curvature).
* Calculates and visualizes the cumulative explained variance.
* Reconstructs the yield curve using the most significant principal components.

## Data Source

The data used in this project consists of daily US Treasury constant maturity rates for 11 tenors (1-month to 30-years). It is obtained from the Federal Reserve Economic Data (FRED) database via the **Quandl API**.

* **Quandl Codes:**
    * `FRED/DGS1MO` (1 Month)
    * `FRED/DGS3MO` (3 Month)
    * `FRED/DGS6MO` (6 Month)
    * `FRED/DGS1` (1 Year)
    * `FRED/DGS2` (2 Year)
    * `FRED/DGS3` (3 Year)
    * `FRED/DGS5` (5 Year)
    * `FRED/DGS7` (7 Year)
    * `FRED/DGS10` (10 Year)
    * `FRED/DGS20` (20 Year)
    * `FRED/DGS30` (30 Year)

**Note:** You will need a free Quandl API key to fetch the data.

## Installation

1.  **Install required libraries:**
    It's recommended to use a virtual environment.
    ```bash
    pip install quandl pandas numpy matplotlib seaborn scikit-learn
    ```
    *(Alternatively, if a `requirements.txt` file is provided: `pip install -r requirements.txt`)*

2.  **Set up Quandl API Key:**
    You need to obtain an API key from [Quandl](https://data.nasdaq.com/sign-up) by registering for a free account. Once you have the key, you will need to insert it into the script where indicated.

## Usage

The analysis follows these main steps:

1.  **Import Libraries and Configure API Key:**
    ```python
    import quandl
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    pd.set_option('display.width', 100)
    pd.set_option('display.max_rows', 500)


    # --- IMPORTANT ---
    # Replace 'YOUR_API_KEY' with your actual Quandl API key
    quandl.ApiConfig.api_key = 'YOUR_API_KEY'
    ```

2.  **Fetch Data:**
    ```python
    treasury_codes = ['FRED/DGS1MO', 'FRED/DGS3MO', 'FRED/DGS6MO', 'FRED/DGS1',
                      'FRED/DGS2', 'FRED/DGS3', 'FRED/DGS5', 'FRED/DGS7',
                      'FRED/DGS10', 'FRED/DGS20', 'FRED/DGS30']

    treasury_df = quandl.get(treasury_codes)

    # Assign meaningful column names
    treasury_df.columns = ['TRESY1mo', 'TRESY3mo', 'TRESY6mo', 'TRESY1y',
                           'TRESY2y', 'TRESY3y', 'TRESY5y', 'TRESY7y',
                           'TRESY10y', 'TRESY20y', 'TRESY30y']

    dataset = treasury_df.copy() # Keep original safe
    ```

3.  **Exploratory Data Analysis (EDA):**
    ```python
    # View last few data points
    print(dataset.tail())

    # Check data types
    print(dataset.dtypes)

    # Get descriptive statistics
    print(dataset.describe())

    # Plot the raw yield curves over time
    dataset.plot(figsize=(16,8))
    plt.title("US Treasury Yield Curves Over Time")
    plt.ylabel("Rate (%)")
    plt.legend(bbox_to_anchor=(1.01, 0.9), loc=2)
    plt.show()

    # Calculate and plot the correlation matrix
    correlation = dataset.corr()
    plt.figure(figsize=(12,10))
    plt.title('Correlation Matrix of Treasury Yields')
    sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix', fmt='.2f')
    plt.show()
    ```

4.  **Data Preparation:**
    ```python
    # Check for missing values
    print('Contains Null Values Before Imputation:', dataset.isnull().values.any())

    # Fill missing values using forward fill (propagate last valid observation forward)
    dataset.fillna(method='ffill', inplace=True)

    # Drop any remaining rows with NA (e.g., at the beginning of the series)
    dataset.dropna(axis=0, inplace=True)
    print('Contains Null Values After Cleaning:', dataset.isnull().values.any())


    # Standardize the data (mean=0, variance=1)
    scaler = StandardScaler()
    rescaledDataset = pd.DataFrame(scaler.fit_transform(dataset),
                                   columns=dataset.columns,
                                   index=dataset.index)

    # Plot scaled data
    rescaledDataset.plot(figsize=(16,8))
    plt.title("Standardized US Treasury Yield Curves")
    plt.ylabel("Standardized Rate")
    plt.legend(bbox_to_anchor=(1.01, 0.9), loc=2)
    plt.show()
    ```

5.  **Principal Component Analysis (PCA):**
    ```python
    # Fit PCA model
    pca = PCA()
    pca.fit(rescaledDataset)

    # Analyze explained variance
    NumEigenvalues = 5 # Or len(pca.explained_variance_ratio_) for all
    fig, axes = plt.subplots(ncols=2, figsize=(14,4))
    pd.Series(pca.explained_variance_ratio_[:NumEigenvalues]).sort_values().plot.barh(
        title='Explained Variance Ratio by Top Factors', ax=axes[0])
    pd.Series(pca.explained_variance_ratio_[:NumEigenvalues]).cumsum().plot(
        ylim=(0,1), ax=axes[1], title='Cumulative Explained Variance')
    plt.show()

    # Display cumulative variance table
    print("Cumulative Explained Variance by Top Components:")
    print(pd.Series(np.cumsum(pca.explained_variance_ratio_)).to_frame(
        'Cumulative Explained Variance').head(NumEigenvalues).style.format('{:,.2%}'.format))
    ```

6.  **Interpret Principal Components:**
    ```python
    # Function to calculate weights (loadings)
    def PCWeights(pca_model, data_columns):
        weights = pd.DataFrame(pca_model.components_, columns=data_columns)
        # Optional: Normalize weights for easier comparison (sum to 1 per component)
        # normalized_weights = weights.apply(lambda x: x / x.abs().sum(), axis=1)
        return weights # or normalized_weights

    weights = PCWeights(pca, dataset.columns)

    NumComponents = 3 # Focus on the top 3
    topPortfolios = weights[:NumComponents]
    topPortfolios.index = [f'Principal Component {i}' for i in range(1, NumComponents + 1)]

    # Plot component weights (loadings) across maturities
    axes = topPortfolios.T.plot.bar(subplots=True, legend=False, figsize=(14, 10),
                                     title=[f'PC {i} Loadings' for i in range(1, NumComponents + 1)])
    plt.subplots_adjust(hspace=0.4)
    # axes[0].set_ylim(0, .2) # Adjust ylim based on observed weights if needed
    plt.show()

    # Plot loadings as lines for Level, Slope, Curvature interpretation
    plt.figure(figsize=(10, 7))
    plt.plot(pca.components_[0], marker='o', label='PC1 (Level)')
    plt.plot(pca.components_[1], marker='o', label='PC2 (Slope)')
    plt.plot(pca.components_[2], marker='o', label='PC3 (Curvature)')
    plt.xticks(ticks=range(len(dataset.columns)), labels=dataset.columns, rotation=45)
    plt.xlabel("Maturity")
    plt.ylabel("Loading")
    plt.title("Principal Component Loadings (Level, Slope, Curvature)")
    plt.legend()
    plt.grid(True)
    plt.show()
    ```

7.  **Reconstruct Data using Principal Components:**
    ```python
    nComp = 3 # Number of components for reconstruction
    # Transform data into PCA space (get component scores)
    pca_scores = pca.transform(rescaledDataset)

    # Reconstruct data using the first nComp components
    reconstructed_scaled = np.dot(pca_scores[:, :nComp], pca.components_[:nComp, :])

    # Inverse transform to original scale
    reconstructed_original = scaler.inverse_transform(reconstructed_scaled)
    reconstructed_df = pd.DataFrame(reconstructed_original,
                                    columns=dataset.columns,
                                    index=dataset.index)

    # Plot the reconstructed yield curves
    plt.figure(figsize=(16, 8))
    plt.plot(reconstructed_df)
    plt.ylabel("Reconstructed Treasury Rate (%)")
    plt.title(f"Yield Curve Reconstructed using {nComp} Principal Components")
    # plt.legend(bbox_to_anchor=(1.01, 0.9), loc=2) # Can be crowded, optional
    plt.show()

    # Optional: Compare a single day's original vs reconstructed curve
    # sample_date = '2022-07-15' # Example date
    # if sample_date in dataset.index:
    #     plt.figure(figsize=(10, 6))
    #     original_curve = dataset.loc[sample_date]
    #     reconstructed_curve = reconstructed_df.loc[sample_date]
    #     maturities = [0.08, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30] # Approx years for plotting x-axis
    #
    #     plt.plot(maturities, original_curve.values, marker='o', linestyle='-', label='Original')
    #     plt.plot(maturities, reconstructed_curve.values, marker='x', linestyle='--', label=f'Reconstructed ({nComp} PCs)')
    #     plt.title(f'Original vs Reconstructed Yield Curve for {sample_date}')
    #     plt.xlabel('Maturity (Years)')
    #     plt.ylabel('Rate (%)')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    ```

## Results

* The analysis reveals a high degree of correlation between Treasury yields of different maturities.
* PCA effectively reduces the dimensionality of the yield curve data. The first three principal components (PCs) capture over **99.5%** of the total variance in the yield curve movements.
    * **PC1 (~84.5% variance):** Represents parallel shifts in the yield curve (**Level**). Loadings are positive and relatively constant across all maturities.
    * **PC2 (~14.1% variance, Cumulative ~98.6%):** Represents changes in the slope of the yield curve (**Slope**). Loadings typically transition from negative for short maturities to positive for long maturities (or vice-versa).
    * **PC3 (~1.1% variance, Cumulative ~99.7%):** Represents changes in the curvature of the yield curve (**Curvature**). Loadings often show a 'U' or inverted 'U' shape, impacting mid-term maturities differently than short and long-term ones.
* Reconstructing the yield curve data using only these first three principal components provides a very close approximation to the original dataset, demonstrating the efficiency of PCA for capturing the essential dynamics.
* This implies that hedging strategies could potentially focus primarily on managing exposure to these three factors (level, slope, curvature) rather than individual maturities.

## Visualizations

The script generates several plots to aid understanding:

* Time series of raw Treasury yields.
* Correlation matrix heatmap.
* Time series of standardized Treasury yields.
* Explained variance ratio bar chart.
* Cumulative explained variance line plot.
* Bar plots of loadings for the top principal components.
* Line plot comparing the loadings of PC1, PC2, and PC3 (Level, Slope, Curvature interpretation).
* Time series of reconstructed Treasury yields using the top 3 PCs.
* (Optional) Comparison plot of original vs. reconstructed yield curve for a specific date.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for any bugs, suggestions, or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (assuming you add an MIT license file).

## Acknowledgements

This project structure and analysis approach were inspired by the work and examples demonstrated by **alecontuIT**. The data is sourced from FRED via the Quandl API.
