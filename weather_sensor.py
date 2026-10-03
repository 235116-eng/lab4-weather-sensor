import pandas as pd
import numpy as np
from scipy import stats

data = {
    "Hour": range(1, 16),
    "Temperature": [
        22, 23, np.nan, np.nan, np.nan,
        25, 26, 58, 24, 23,
        22, 21, 23, 24, 25
    ]
}

df = pd.DataFrame(data)

print("Stats BEFORE cleaning:")

raw_temperature = df["Temperature"]

print("Mean:", raw_temperature.mean())
print("Std:", raw_temperature.std())

valid_temperature = raw_temperature.dropna()

print("Skewness:", stats.skew(valid_temperature))
print("Kurtosis:", stats.kurtosis(valid_temperature))

df["Temperature_Clean"] = (
    df["Temperature"].interpolate(method="linear")
)

print("\nAfter interpolation, missing values:",
      df["Temperature_Clean"].isnull().sum())

z_scores = stats.zscore(df["Temperature_Clean"])

outlier_mask = np.abs(z_scores) > 2

outlier_rows = df[outlier_mask]

print("\nZ-score(>2) outlier detection:")

if len(outlier_rows) > 0:
    for index in outlier_rows.index:
        print(
            "Outlier detected at Hour:",
            df.loc[index, "Hour"],
            "(value:",
            df.loc[index, "Temperature_Clean"],
            ")"
        )
else:
    print("No outlier detected.")

median_temperature = df["Temperature_Clean"].median()

df.loc[outlier_mask, "Temperature_Clean"] = median_temperature

print("Replaced outlier with column median:",
      median_temperature)

df["Temperature_Zscore"] = stats.zscore(
    df["Temperature_Clean"]
)

comparison = df[
    [
        "Hour",
        "Temperature",
        "Temperature_Clean",
        "Temperature_Zscore"
    ]
]

print("\nBefore/After comparison:")
print(comparison)
clean_mean = df["Temperature_Clean"].mean()
clean_std = df["Temperature_Clean"].std()

print("\nStats AFTER cleaning:")
print("Mean:", clean_mean)
print("Std:", clean_std)

print("\nComparison:")
print("Mean BEFORE:", raw_temperature.mean())
print("Std BEFORE:", raw_temperature.std())
print("Mean AFTER:", clean_mean)
print("Std AFTER:", clean_std)