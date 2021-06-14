import toml

def read_in_surr_config(exp_config_file):
    experiment_config = toml.load(exp_config_file)
    elite_map_config = toml.load(experiment_config["Search"]["ConfigFilename"])
    return experiment_config, elite_map_config



def get_label(experiment_config):
    legend = ""
    if experiment_config["Search"]["Category"] == "Surrogated":
        if experiment_config["Search"]["Type"] == "MAP-Elites":
            if experiment_config["Surrogate"]["Type"] == "FullyConnectedNN":
                legend += "FCNN" + " DSA-ME"
            elif experiment_config["Surrogate"]["Type"] == "DeepSetModel":
                legend += "Deep-set" + " DSA-ME"
            elif experiment_config["Surrogate"]["Type"] == "LinearModel":
                legend += "Linear" + " DSA-ME"
            else:
                legend += experiment_config["Surrogate"]["Type"] + " DSA-ME"
    elif experiment_config["Search"]["Category"] == "Distributed":
        legend += experiment_config["Search"]["Type"]
    return legend