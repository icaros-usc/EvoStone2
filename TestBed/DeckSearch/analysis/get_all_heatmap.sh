step_length="$1"

declare -a all_base_exp=(
    "logs/classic_miracle_rogue_strat_deck/to_plot/2021-07-14_21-15-31_Distributed_MAP-Elites_Classic_Miracle_Rogue_show"
    "logs/classic_miracle_rogue_strat_deck/to_plot/2021-07-14_21-15-29_Surrogated_MAP-Elites_FullyConnectedNN_Classic_Miracle_Rogue_show"
    "logs/classic_miracle_rogue_strat_deck/to_plot/2021-07-14_21-15-30_Surrogated_MAP-Elites_LinearModel_Classic_Miracle_Rogue_RCA_show"
    "logs/classic_miracle_rogue_strat_deck/to_plot/2021-07-31_15-43-33_Surrogated_MAP-Elites_FixedFCNN_show"
    )

declare -a all_more_exp=(
    "logs/classic_miracle_rogue_strat_deck_more_target/to_plot/2021-07-28_23-29-14_Surrogated_MAP-Elites_LinearModel_show"
    "logs/classic_miracle_rogue_strat_deck_more_target/to_plot/2021-07-29_14-00-59_Surrogated_MAP-Elites_FullyConnectedNN_Classic_Miracle_Rogue_show"
    "logs/classic_miracle_rogue_strat_deck_more_target/to_plot/2021-07-29_17-05-44_Surrogated_MAP-Elites_FixedFCNN_Classic_Miracle_Rogue_show"
)


for log_dir in "${all_base_exp[@]}"
do
    echo "Log Dir: ${log_dir}"
    python analysis/gen_metrics.py -l "${log_dir}" -s $step_length
    cp $log_dir/metrics/elites_archive/heatmap/*.pdf ~/default_targets/
done


for log_dir in "${all_more_exp[@]}"
do
    echo "Log Dir: ${log_dir}"
    python analysis/gen_metrics.py -l "${log_dir}" -s $step_length
    cp $log_dir/metrics/elites_archive/heatmap/*.pdf ~/more_targets/
done
