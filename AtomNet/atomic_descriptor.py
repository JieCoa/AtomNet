from mendeleev import element
from mendeleev.fetch import fetch_table, fetch_ionization_energies
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import time
import json
import os


def get_available_filename(len_of_features):
    """
    Find available file names in the directory `base_dir` based on the number of features `num_features`.
    If a base file exists, attempt to update it with names such as update01, update02, and so on, incrementing sequentially.
    """
    base_filename = f'atom_features({len_of_features}d)'
    filepath = os.path.join(base_filename, '.csv')

    if not os.path.exists(filepath):
        return base_filename

    i = 1
    while True:
        suffix = f'_update{i:02d}'
        new_filename = f'atom_features({len_of_features}d){suffix}'
        new_filepath = os.path.join(new_filename)
        if not os.path.exists(new_filepath, '.csv'):
            return new_filepath
        i += 1


def get_data_by_features(atom_range=[1, 117], atom_properties=[], atom_methods=[], public_features=[], **kwargs):
    if atom_range[0] < 1 or atom_range[1] > 117:
        raise ValueError('atom_range must be in [1, 117]')

    df = fetch_table('elements')
    features = []
    print("Starting to fetch data for atom properties and methods...")
    start = time.perf_counter()

    print("Fetching data for atom properties...")
    for prop in atom_properties:
        cur_list = []
        for i in range(atom_range[0], atom_range[1] + 1):
            # element(i) obtains the element object through the atomic number, noting that it starts from 1
            cur_list.append(element(i).__getattribute__(prop))
        newSeries = pd.Series(cur_list, name=prop)
        features.append(newSeries)
    print("Finish fetching data for atom properties!")

    print("Fetching data for atom methods...")
    for method in atom_methods:
        cur_list = []
        for i in range(atom_range[0], atom_range[1] + 1):
            if method == 'oxidation_states':
                mx = element(i).__getattribute__(method)('main')
                if len(mx) > 0:
                    mx = max(map(abs, mx))
                else:
                    mx = np.nan
                cur_list.append(mx)
            else:
                cur_list.append(element(i).__getattribute__(method)())
        newSeries = pd.Series(cur_list, name=method)
        features.append(newSeries)
    print("Finish fetching data for atom methods!")

    print("Fetching data for public features...")
    for attr in public_features:
        newSeries = pd.Series(df[attr], name=attr)
        # newSeries.index = range(1, len(newSeries) + 1)
        sub_newSeries = newSeries[(atom_range[0] - 1):(atom_range[1] - 1)]
        features.append(sub_newSeries)
    print("Finish fetching data for public features!")

    df_features = pd.concat(features, axis=1)
    print(df_features)
    print(f"It takes {time.perf_counter() - start} s to obtain features")
    print('-' * 50)
    return df_features


def log_transform(features):
    return np.log(features)


def KBins(features, n_bins=10, encode='ordinal', strategy='kmeans'):
    """ Perform binning for continuous variables
    :param encode: ‘onehot’ - One-hot encoding, ‘ordinal’ - Ordinal encoding (currently, only 'ordinal' can be used)
        - If it is ordinal encoding, the result is a single column with a shape of (n_valid,).
        - If it is onehot encoding, the result will be multiple columns with a shape of (n_valid, n_bins).
    :param strategy: 'uniform' - equal-width binning, 'quantile' - equal-depth binning, 'kmeans' - kmeans binning

    Instructions:
    1. The data input to KBinsDiscretizer requires 2 dimensions (reshape(-1, 1)).
    2. Use ~feature.isna() to create a boolean mask that distinguishes between NaN and non-NaN values.
    """
    if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame(features)

    numeric_kbin = []
    for col in features.columns:
        feature = features[col]
        valid_mask = ~feature.isna()
        valid_feature = feature[valid_mask].values.reshape(-1, 1)

        if len(valid_feature) == 0:
            numeric_kbin.append(pd.Series(feature, name=col))
            continue

        dis = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
        binned_values = dis.fit_transform(valid_feature)

        res = pd.Series(index=feature.index, dtype=float)
        res[valid_mask] = binned_values.ravel()
        res[~valid_mask] = np.nan
        numeric_kbin.append(res.rename(col))

    res = pd.concat(numeric_kbin, axis=1)
    # print(res)
    return res


def cate_colName(Transformer, category_cols, drop='if_binary'):
    """
    Generate new one-hot encoded feature names based on the field names after one-hot encoding of discrete fields in Transformer

    :param Transformer: One-hot encoding converter
    :param category_cols: The discrete variable input to the converter must be an iterable object, such as a list
    :param drop: Parameter of the one-hot encoder converter
    """

    cate_cols_new = []
    col_value = Transformer.categories_
    # print(col_value[0])  # list

    for i, j in enumerate(category_cols):
        if (drop == 'if_binary') & (len(col_value[i]) == 2):
            cate_cols_new.append(j)
        else:
            if np.issubdtype(type(col_value[0][i]), np.number):
                for f in col_value[i]:
                    f = int(f)
                    feature_name = j + '_' + str(f)
                    cate_cols_new.append(feature_name)
            else:
                for f in col_value[i]:
                    feature_name = j + '_' + f
                    cate_cols_new.append(feature_name)
    return cate_cols_new


def oneHotEncoderFeatures(features=[], drop='if_binary'):
    one_hot = []
    for col in features.columns:
        category = features[col].dropna().unique()

        if np.issubdtype(features[col].dtype, np.number):
            categories = sorted(category)
        else:
            categories = list(category)

        encoder = OneHotEncoder(categories=[categories], drop=drop, handle_unknown='ignore')
        onehot = encoder.fit_transform(features[[col]])
        df = pd.DataFrame(onehot.toarray(), columns=cate_colName(encoder, [col], drop=drop)).astype(int)
        # print(df)
        one_hot.append(df)

    one_hot = pd.concat(one_hot, axis=1)
    return one_hot


if __name__ == '__main__':
    """ 
    Feature of 'Version 1':
    
        atom_properties = ['period', 'atomic_volume', 'melting_point']
        atom_methods = ['electronegativity_sanderson', 'nvalence']
        public_features = ['group_id', 'covalent_radius_cordero', 'electron_affinity', 'block', 'atomic_number',
                           'metallic_radius', 'lattice_constant', 'atomic_radius_rahm']
        Ionization_energy_features = ['Ionization energy']
        category_cols = ['period', 'group_id', 'block', 'nvalence']

    
    Feature of 'Version 2': atom_features(106d)
        
        atom_properties = ['period', 'c6']
        atom_methods = ['electronegativity_sanderson', 'nvalence']
        public_features = ['group_id', 'electron_affinity', 'block', 'lattice_constant', 'atomic_radius_rahm']
        Ionization_energy_features = ['Ionization energy']
        category_cols = ['period', 'group_id', 'block', 'nvalence']
        
    
    Feature of 'Version 3': atom_features(116d), with 'atomic_volume' added compared to 'Version 2'
        
        atom_properties = ['period', 'atomic_volume', 'c6']
        atom_methods = ['electronegativity_sanderson', 'nvalence']
        public_features = ['group_id', 'electron_affinity', 'block', 'lattice_constant', 'atomic_radius_rahm']
        Ionization_energy_features = ['Ionization energy']
        category_cols = ['period', 'group_id', 'block', 'nvalence']

    
    Feature of 'Version 4'(AtomNet's atomic descriptor module): atom_features(116d)_update01. Compared to 'Version 3', 
    atomic radius has been changed from 'atomic_radius_rahm' to 'atomic_radius'
        
        atom_properties = ['period', 'atomic_volume', 'c6']
        atom_methods = ['electronegativity_sanderson', 'nvalence']
        public_features = ['group_id', 'electron_affinity', 'block', 'lattice_constant', 'atomic_radius']
        Ionization_energy_features = ['Ionization energy']
        category_cols = ['period', 'group_id', 'block', 'nvalence']
    
    
    Features of 'Version 5': Refer to https://arxiv.org/html/2503.04492v1
        
        atom_properties = ['period', 'atomic_volume', 'c6']
        atom_methods = ['electronegativity', 'oxidation_states']
        public_features = ['electron_affinity', 'block', 'lattice_constant', 'vdw_radius']
        Ionization_energy_features = ['Ionization energy']
        category_cols = ['period', 'block', 'oxidation_states']
    """

    # 'Version 4'(AtomNet's atomic descriptor module)
    atom_properties = ['period', 'atomic_volume', 'c6']
    atom_methods = ['electronegativity_sanderson', 'nvalence']
    public_features = ['group_id', 'electron_affinity', 'block', 'lattice_constant', 'atomic_radius']
    Ionization_energy_features = ['Ionization energy']
    category_cols = ['period', 'group_id', 'block', 'nvalence']

    # 1. Call `get_data_by_features` to retrieve the target features.
    all_features = get_data_by_features(atom_properties=atom_properties, atom_methods=atom_methods, public_features=public_features)

    # 2. To obtain the ionization energy feature, you need to call the `fetch_ionization_energies()` function.
    if Ionization_energy_features is not None:
        # The DataFrame obtained by fetch_ionization_energies() has no index and contains only two columns:
        # 'atomic_number' and 'IE1', where 'IE1' is the first ionization energy.

        fea = fetch_ionization_energies().reset_index().drop('atomic_number', axis=1)
        fea = fea.rename(columns={'IE1': 'Ionization energy'})
        all_features = pd.concat([all_features, fea], axis=1)
        print(all_features)

    # 3. Divide features into discrete features and continuous features.
    category_features = all_features[category_cols]
    numeric_features = all_features.drop(category_cols, axis=1)

    if 'atomic_number' in numeric_features.columns:
        numeric_features = numeric_features.drop('atomic_number', axis=1)
    if 'atomic_number' in public_features:
        assert len(category_features.columns) + len(numeric_features.columns) == len(all_features.columns) - 1
    else:
        assert len(category_features.columns) + len(numeric_features.columns) == len(all_features.columns)

    # print(category_features)
    # print(numeric_features)

    for col in category_features.columns:
        print(category_features[col].name)
        print(category_features[col].dropna().unique())
    print(category_features.nunique(dropna=False))

    print('-' * 50)

    print(numeric_features.nunique(dropna=False))
    print(numeric_features.describe())

    print('-' * 50)
    # print(category_features.isnull().sum())
    # print(numeric_features.isnull().sum())

    # log_scale_features = ['atomic_volume', 'melting_point', 'covalent_radius_cordero', 'metallic_radius',
    # 'atomic_radius_rahm', 'Ionization energy']
    log_scale_features = []
    for col in log_scale_features:
        numeric_features[col] = log_transform(numeric_features[col])

    if len(log_scale_features) > 0:
        print(numeric_features.describe())

    kbins_numeric_features = KBins(numeric_features, n_bins=20)

    cat_oneHotFeatures = oneHotEncoderFeatures(category_features, drop='if_binary')
    num_oneHotFeatures = oneHotEncoderFeatures(kbins_numeric_features, drop='if_binary')

    new_atom_oneHotFeatures = pd.concat([cat_oneHotFeatures, num_oneHotFeatures], axis=1)
    print(new_atom_oneHotFeatures)
    print(f"The dimension of the generated Atomic Descriptor is: {len(new_atom_oneHotFeatures.columns)}.")
    # print(cat_oneHotFeatures)
    # print('-' * 50)
    # print(num_oneHotFeatures)

    file_path = get_available_filename(len(new_atom_oneHotFeatures.columns))
    new_atom_oneHotFeatures.to_csv(f'{file_path}.csv',
                                   encoding='utf-8',
                                   index=True)

    new_atom_features_dict = {}
    for idx, row in new_atom_oneHotFeatures.iterrows():
        key = idx + 1
        value = row.tolist()
        new_atom_features_dict[str(key)] = value

    with open(f'{file_path}.json', 'w') as f:
        json.dump(new_atom_features_dict, f, indent=4)

    print(f'{file_path}.json')
