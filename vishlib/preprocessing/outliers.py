import pandas as pd
import numpy as np
import stats
from typing import Union, Set, List, Tuple


def remove_outliers(df: pd.DataFrame,
                    method: str = 'iqr',
                    iqr_factor: float = 1.5,
                    features: Union[List[str], Set[str]] = None,
                    axis: int = 1,
                    ignore_nan: bool = True,
                    ) -> pd.DataFrame:

    if features is None:
        features = df.columns.to_list()
    else:
        features = list(features)

    holdout_features = list(set(df.columns) - set(features))
    holdout_df = df[holdout_features].copy()

    if method == 'z-score':
        
        if axis == 1:

            df_wout_outliers = pd.DataFrame()
            
            for feature_name_iter in features:
                feature = df[feature_name_iter]
                feature_zscore = (feature - feature.mean()) / feature.std()
                df_wout_outliers[feature_name_iter] = feature.where(feature_zscore.abs() < 3)
            
            df_wout_outliers = pd.concat([holdout_df, df_wout_outliers], axis=1)
            return df_wout_outliers

        elif axis == 0:

            features_df = df[features]
            
            if not ignore_nan:
                features_df_zscore = ((features_df - features_df.mean()) / features_df.std(ddof = 0)).abs()
                features_df_zscore.replace(to_replace = {np.nan: 0}, inplace = True)
                df_wout_outliers = df[(features_df_zscore <= 3).all(axis = 1)].reset_index(drop = True)
            else:
                z_scores = stats.zscore(frame)
                abs_z_scores = np.abs(z_scores)
                df_filter = (abs_z_scores < 3).all(axis=1)
                df_wout_outliers = df[df_filter]

            return df_wout_outliers

    if method == 'iqr':
        
        if axis == 1:

            df_wout_outliers = pd.DataFrame()

            for feature in features:
                series = df[feature]
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = abs(q1 - q3)
                series_without_outliers_keep_index = series[~ ((series < (q1 - iqr_factor * iqr)) | (series > (q3 + iqr_factor * iqr)))]
                df_wout_outliers = pd.concat([df_wout_outliers, series_without_outliers_keep_index], axis = 1)

            for feature in list(set(df.columns) - set(features)):
                series = df[feature]
                df_wout_outliers = pd.concat([df_wout_outliers, series], axis = 1)

            return df_wout_outliers
        
        else:

            subdf_to_filter = df[features]
            q1 = subdf_to_filter.quantile(0.25)
            q3 = subdf_to_filter.quantile(0.75)
            iqr = (q1 - q3).abs()
            df_wout_outliers = df[~ ((subdf_to_filter < (q1 - iqr_factor*iqr)) | (subdf_to_filter > (q3 + iqr_factor*iqr))).any(axis = 1)]

            return df_wout_outliers
