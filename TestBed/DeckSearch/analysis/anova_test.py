import ast
import argparse
import pandas as pd
import numpy as np
import scipy.stats as st

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f',
        '--numerical_measure_path',
        help='Path to to numerical_measure.csv',
        required=True,
    )
    alpha = 0.05
    opt = parser.parse_args()
    nm_path = opt.numerical_measure_path
    numerical_measure_df = pd.read_csv(nm_path, index_col=0)
    all_algos = numerical_measure_df.columns.values
    for measure, data in numerical_measure_df.iterrows():
        print(f"Current metrics: {measure}")
        data_list = [ast.literal_eval(data_str) for data_str in data]
        f_stat, p_val = st.f_oneway(*data_list)
        print("ANOVA one way test")
        if p_val < alpha:
            text = "Reject"
        else:
            text = "No Reject"
        print(f"F stats = {f_stat}, p value = {p_val}, {text}\n")

        num_pairs = len(all_algos) * (len(all_algos) - 1) / 2
        for i in range(0, len(all_algos)):
            for j in range(i + 1, len(all_algos)):
                algo1 = all_algos[i]
                algo2 = all_algos[j]
                algo1_data = ast.literal_eval(data[algo1])
                algo2_data = ast.literal_eval(data[algo2])
                pair_f_stat, pair_p_val = st.ttest_ind(algo1_data, algo2_data)
                if pair_p_val < alpha / num_pairs:
                    text = "Reject"
                else:
                    text = "No Reject"
                print(f"T Test of \"{algo1}\" and \"{algo2}\"")
                print(
                    f"F stats = {pair_f_stat}, p value = {pair_p_val}, {text}")

        print("\n")