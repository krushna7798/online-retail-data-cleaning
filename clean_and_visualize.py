import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset():
    """Load the dataset whether it is saved as .csv or .xlsx."""
    if os.path.exists("online_retail_II.csv"):
        print("--> Loading 'online_retail_II.csv'...")
        return pd.read_csv("online_retail_II.csv")
    elif os.path.exists("online_retail_II.xlsx"):
        print("--> Loading 'online_retail_II.xlsx'...")
        return pd.read_excel("online_retail_II.xlsx")
    else:
        raise FileNotFoundError(
            "Dataset not found! Ensure 'online_retail_II.csv' or 'online_retail_II.xlsx' "
            "is in the same directory as this script."
        )


def cap_outliers_iqr(series):
    """Cap extreme numeric outliers using the Interquartile Range (IQR) technique."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = max(0.0, q1 - 1.5 * iqr)
    upper_bound = q3 + 1.5 * iqr
    return np.clip(series, lower_bound, upper_bound)


def main():
    print("=" * 60)
    print("ONLINE RETAIL II - DATA CLEANING & VISUALIZATION PIPELINE")
    print("=" * 60)

    # -------------------------------------------------------------
    # 1. LOAD & INSPECT
    # -------------------------------------------------------------
    df = load_dataset()
    print(f"Initial Dataset Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # -------------------------------------------------------------
    # 2. DATA CLEANING
    # -------------------------------------------------------------
    print("\n[Step 1/4] Cleaning data...")

    # Drop exact duplicate rows
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f" -> Removed {initial_rows - len(df):,} duplicate rows.")

    # Remove cancellations (Invoices starting with 'C' or negative quantities)
    df["Invoice"] = df["Invoice"].astype(str)
    df = df[~df["Invoice"].str.startswith("C")]
    df = df[df["Quantity"] > 0]

    # Filter out zero or negative prices
    df = df[df["Price"] > 0]

    # Handle missing values
    df["Description"] = df["Description"].fillna("Unknown")
    df["Customer ID"] = df["Customer ID"].fillna(0).astype(int)

    # Convert dates to standard datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Feature Engineering: Total Sales per transaction
    df["Total_Sales"] = df["Quantity"] * df["Price"]

    # -------------------------------------------------------------
    # 3. OUTLIER HANDLING
    # -------------------------------------------------------------
    print("\n[Step 2/4] Applying IQR outlier capping...")
    df["Quantity_Capped"] = cap_outliers_iqr(df["Quantity"])
    df["Total_Sales_Capped"] = cap_outliers_iqr(df["Total_Sales"])

    print(f"Cleaned Dataset Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # -------------------------------------------------------------
    # 4. EXPORT CLEANED DATASET
    # -------------------------------------------------------------
    output_csv = "cleaned_online_retail_II.csv"
    print(f"\n[Step 3/4] Exporting cleaned data to '{output_csv}'...")
    df.to_csv(output_csv, index=False)
    print(" -> CSV export complete.")

    # -------------------------------------------------------------
    # 5. VISUALIZATIONS & REPORTING
    # -------------------------------------------------------------
    print("\n[Step 4/4] Generating visual charts...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Chart 1: Top 10 Best-Selling Products by Volume
    top_products = (
        df.groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
    )
    top_products = top_products[top_products.index != "Unknown"].head(10)
    sns.barplot(
        ax=axes[0, 0],
        x=top_products.values,
        y=top_products.index,
        hue=top_products.index,
        palette="viridis",
        legend=False
    )
    axes[0, 0].set_title("Top 10 Best-Selling Products by Quantity", fontsize=12, pad=10)
    axes[0, 0].set_xlabel("Total Units Sold")
    axes[0, 0].set_ylabel("Product Name")

    # Chart 2: Monthly Revenue Trend
    monthly_sales = df.set_index("InvoiceDate").resample("ME")["Total_Sales"].sum()
    axes[0, 1].plot(
        monthly_sales.index,
        monthly_sales.values,
        color="#1f77b4",
        marker="o",
        linewidth=2
    )
    axes[0, 1].set_title("Monthly Revenue Trajectory", fontsize=12, pad=10)
    axes[0, 1].set_ylabel("Total Sales (£)")
    axes[0, 1].set_xlabel("Invoice Date")
    axes[0, 1].tick_params(axis="x", rotation=30)

    # Chart 3: Quantity Distribution (Outlier Visual Verification)
    sns.boxplot(ax=axes[1, 0], x=df["Quantity_Capped"], color="#ff7f0e")
    axes[1, 0].set_title("Distribution of Units Sold (Post-IQR Capping)", fontsize=12, pad=10)
    axes[1, 0].set_xlabel("Quantity per Transaction")

    # Chart 4: Top 5 International Markets (Excluding UK)
    top_countries = (
        df[df["Country"] != "United Kingdom"]
        .groupby("Country")["Total_Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )
    sns.barplot(
        ax=axes[1, 1],
        x=top_countries.index,
        y=top_countries.values,
        hue=top_countries.index,
        palette="mako",
        legend=False
    )
    axes[1, 1].set_title("Top 5 International Markets by Revenue (Excl. UK)", fontsize=12, pad=10)
    axes[1, 1].set_ylabel("Total Revenue (£)")
    axes[1, 1].set_xlabel("Country")
    axes[1, 1].tick_params(axis="x", rotation=20)

    # Save visual dashboard
    output_image = "retail_eda_report.png"
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f" -> Visualization report saved as '{output_image}'.")
    plt.show()

    print("\nPipeline executed successfully!")


if __name__ == "__main__":
    main()