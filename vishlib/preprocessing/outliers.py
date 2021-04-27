import pandas as pd
import numpy as np
import stats
from typing import Union, Set, List, Tuple


def remove_outliers(df: pd.DataFrame,
                    method: str = 'iqr',
                    iqr_factor: float = 1.5,
                    features_to_analyze: Union[List[str], Set[str]] = None,
                    by_feature: bool = False,
                    with_nans: bool = False,
                    return_num_removed: bool = False,
                    # plot_outlier_boundaries: bool = False,
                    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, int]]:
    """
    
    :param df:
    :param method:
    :param iqr_factor:
    :param features_to_analyze:
    :param by_feature:
    :param with_nans:
    :param return_num_removed:
    :return:
    """

    # if checking number of removed rows
    size_before_removal = df.shape[0]

    # defaults initialization
    if features_to_analyze is None:
        features_to_analyze = df.columns.to_list()
    else:
        # cast to list, whether set or list
        features_to_analyze = list(features_to_analyze)

    # holdout features in separate dataframe to be concated
    holdout_features = list(set(df.columns) - set(features_to_analyze))
    holdout_dataframe = df[holdout_features].copy()

    # outliers deviate more than 3 sigmas
    if method == 'z-score':

        # replacing outliers with np.nan
        if by_feature:

            # assigning to new df object
            df_without_outliers = pd.DataFrame()

            # replacing outliers with np.nan
            for feature_name_iter in features_to_analyze:
                feature = df[feature_name_iter]
                feature_zscore = (feature - feature.mean()) / feature.std()
                df_without_outliers[feature_name_iter] = feature.where(feature_zscore.abs() < 3)

            # reunite with features not analyzed for outliers
            df_without_outliers = pd.concat([holdout_dataframe, df_without_outliers], axis=1)

            # outcome info
            print('Number of rows removed: ', size_before_removal - df_without_outliers.shape[0])

            # return as specified
            if return_num_removed: return df_without_outliers, size_before_removal - df_without_outliers.shape[0]
            else: return df_without_outliers

        # removing rows
        else:

            # df on which to filter
            frame = df[features_to_analyze]

            # supporting dfs with nans - note: all rows with nans are removed
            if with_nans:

                frame_zscore = ((frame - frame.mean()) / frame.std(ddof = 0)).abs()

                # nans are replaced with zero ONLY IN FILTERING DATAFRAME, so that they don't influence the removal of rows
                frame_zscore.replace(to_replace = {np.nan: 0}, inplace = True)
                df_without_outliers = df[(frame_zscore <= 3).all(axis = 1)].reset_index(drop = True)

            # not supporting nans
            else:

                # filtering with scipy methods
                z_scores = stats.zscore(frame)
                abs_z_scores = np.abs(z_scores)
                filtered_entries = (abs_z_scores < 3).all(axis=1)
                df_without_outliers = df[filtered_entries]

            # outcome info
            print('Number of rows removed: ', size_before_removal - df_without_outliers.shape[0])

            # return as specified
            if return_num_removed: return df_without_outliers, size_before_removal - df_without_outliers.shape[0]
            else: return df_without_outliers

    # outliers deviate 1.5*IQR from 1st and 3rd quartile boundaries
    if method == 'iqr':

        if by_feature:
            df_without_outliers = pd.DataFrame()

            for feature in features_to_analyze:

                series = df[feature]

                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = abs(q1 - q3)

                series_without_outliers_keep_index = series[~ ((series < (q1 - iqr_factor * iqr)) | (series > (q3 + iqr_factor * iqr)))]
                df_without_outliers = pd.concat([df_without_outliers, series_without_outliers_keep_index], axis = 1)

            for feature in list(set(df.columns) - set(features_to_analyze)):

                series = df[feature]
                df_without_outliers = pd.concat([df_without_outliers, series], axis = 1)

            if return_num_removed: return df_without_outliers, size_before_removal - df_without_outliers.shape[0]
            else: return df_without_outliers

        else:

            frame_to_filter = df[features_to_analyze]

            # calculate IQR
            q1 = frame_to_filter.quantile(0.25)
            q3 = frame_to_filter.quantile(0.75)
            iqr = (q1 - q3).abs()

            # remove rows which deviate more
            df_without_outliers = df[~ ((frame_to_filter < (q1 - iqr_factor*iqr)) | (frame_to_filter > (q3 + iqr_factor*iqr))).any(axis = 1)]

            if return_num_removed: return df_without_outliers, size_before_removal - df_without_outliers.shape[0]
            else: return df_without_outliers
