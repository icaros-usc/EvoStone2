﻿using System;
using System.Linq;
using Nett;

using SurrogateModel.Surrogate;

using SabberStoneUtil.Config;

namespace SurrogateModel
{
    class Program
    {
        static void Main(string[] args)
        {
            string expLogDir = args[0];
            string expConfigPath = System.IO.Path.Combine(
                           expLogDir, "experiment_config.tml");

            // init card set
            var config = Toml.ReadFile<Configuration>(expConfigPath);

            if (config.Surrogate != null)
            {
                throw new System.ArgumentException(
                    "Do not train a new surrogate model for DSA-ME.");
            }

            CardReader.Init(config);

            string indsLogPath = System.IO.Path.Combine(
                expLogDir, "individual_log.csv");

            // create model
            string modelType = args[1];
            // configurate surrogate model
            SurrogateBaseModel model = null;
            if (modelType == "DeepSetModel")
            {
                model = new DeepSetModel(
                    log_dir_exp: expLogDir,
                    offline_data_file: indsLogPath);
            }
            else if (modelType == "FullyConnectedNN")
            {
                model = new FullyConnectedNN(
                    log_dir_exp: expLogDir,
                    offline_data_file: indsLogPath);
            }
            else if (modelType == "LinearModel")
            {
                model = new LinearModel(
                    log_dir_exp: expLogDir,
                    offline_data_file: indsLogPath);
            }
            else
            {
                throw new System.ArgumentException(
                    "Invalid model type: {0}.\nMust be one of [DeepSetModel, FullyConnectedNN, LinearModel].", modelType);
            }

            // run offline fit
            model.OfflineFit();
        }
    }
}
