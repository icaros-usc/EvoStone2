import ast
import argparse
import pandas as pd
import numpy as np
import scipy.stats as st
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot


def anova_one_way(numerical_measure_df):
    for measure_name, data in numerical_measure_df.iterrows():
        print(f"Current metrics: {measure_name}")
        data_list = [ast.literal_eval(data_str) for data_str in data]
        f_stat, p_val = st.f_oneway(*data_list)
        print("ANOVA one way test")
        if p_val < alpha:
            text = "Reject"
        else:
            text = "No Reject"
        print(f"F stats = {f_stat}, p value = {p_val}, {text}\n")

        # Post hoc test
        num_pairs = len(all_algos) * (len(all_algos) - 1) / 2
        for i in range(0, len(all_algos)):
            for j in range(i + 1, len(all_algos)):
                algo1 = all_algos[i]
                algo2 = all_algos[j]
                algo1_data = ast.literal_eval(data[algo1])
                algo2_data = ast.literal_eval(data[algo2])
                pair_t_stat, pair_p_val = st.ttest_ind(algo1_data, algo2_data)
                if pair_p_val < alpha / num_pairs:
                    text = "Reject"
                else:
                    text = "No Reject"
                print(f"T Test of \"{algo1}\" and \"{algo2}\"")
                print(
                    f"T stats = {pair_t_stat}, p value = {pair_p_val}, {text}")

        print("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f',
        '--numerical_measure_path',
        help='Path to to numerical_measure.csv',
        required=True,
        nargs='+',
    )
    alpha = 0.05
    opt = parser.parse_args()
    nm_paths = opt.numerical_measure_path
    default_t_path, more_t_path = nm_paths
    d_numerical_measure_df = pd.read_csv(default_t_path, index_col=0)
    m_numerical_measure_df = pd.read_csv(more_t_path, index_col=0)

    all_algos = d_numerical_measure_df.columns.values

    algo_to_idx = {algo: idx for idx, algo in enumerate(all_algos)}

    print("ANOVA two-way test")

    for measure_name, _ in d_numerical_measure_df.iterrows():
        print(f"Current metrics: {measure_name}")
        d_curr_measure = d_numerical_measure_df.loc[measure_name]
        m_curr_measure = m_numerical_measure_df.loc[measure_name]
        curr_metrics = {
            measure_name: [],
            "algo_idx": [],
            "algo": [],
            "target": [],
        }
        for algo in all_algos:
            # if algo == "MAP-Elites":
            #     # print("MAP-Elites is not needed")
            #     continue
            for measure in ast.literal_eval(d_curr_measure[algo]):
                curr_metrics[measure_name].append(measure)
                curr_metrics["algo_idx"].append(algo_to_idx[algo])
                curr_metrics["algo"].append(algo)
                curr_metrics["target"].append("Default")

            for measure in ast.literal_eval(m_curr_measure[algo]):
                curr_metrics[measure_name].append(measure)
                curr_metrics["algo_idx"].append(algo_to_idx[algo])
                curr_metrics["algo"].append(algo)
                curr_metrics["target"].append("More")

        curr_metrics_df = pd.DataFrame(curr_metrics)

        curr_metrics_df[curr_metrics_df["target"] == "Default"].to_csv(
            f"analysis/d_{measure_name}.csv")

        curr_metrics_df[curr_metrics_df["target"] == "More"].to_csv(
            f"analysis/m_{measure_name}.csv")

        # save df
        curr_metrics_df.to_csv(f"analysis/{measure_name}.csv")

        # do two-way anova
        formula = f"{measure_name} ~ C(algo) + C(target) + C(algo):C(target)"
        model = ols(formula, curr_metrics_df).fit()
        aov_table = anova_lm(model, typ=2)
        print(aov_table)
        print()

    print("================================")
    print("Default Targets")
    print("================================")
    anova_one_way(d_numerical_measure_df)
    print("================================")
    print("More Targets")
    print("================================")
    anova_one_way(m_numerical_measure_df)