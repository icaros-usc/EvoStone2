# EvoStone 2.0

This project contains the Hearthstone experiment of the paper **[Deep Surrogate Assisted MAP-Elites for Automated Hearthstone Deckbuilding](https://arxiv.org/abs/2112.03534)**. The code base builds upon [EvoStone](https://github.com/tehqin/EvoStone), which contains Hearthstone experiments for the paper *[Covariance Matrix Adaptation for the Rapid Illumination of Behavior Space](https://arxiv.org/abs/1912.02400)*. The project contains distributed implementations of evolutionary algorithms DSA-ME and all the corresponding surrogate models.

This project is designed to be run on a High-Performance Computing (HPC) cluster and is (mainly) divided into three subprojects `DeckEvaluator` (for running Hearthstone games and collecting data from those games), `DeckSearch` (for running distributed versions of each evolutionary algorithm), and `SurrogateModel` (for training and running surrogate models). EvoStone 2.0 is a unified .NET project and all subprojects can be compiled through a single command.

## Installation
To install the project, you need to install the [.NET Core 3.1](https://dotnet.microsoft.com/download) developer toolkit for your system. You may also need the [NuGet](https://docs.microsoft.com/en-us/nuget/install-nuget-client-tools) client tools for updating dependencies for the project. Included in this project is a `setup.py` script to make installation easier. Users can install Python 3 or follow the commands in the script for installation.

Next you need to compile [SabberStone](https://github.com/HearthSim/SabberStone), the Hearthstone simulator for running games. Follow the instructions on the SabberStone github compile the project into `SabberStone.dll`. Put `SabberStone` in the same root directory as `EvoStone2`. Therefore the project structure should look like the follows:
```
project
|
└───EvoStone2
|
└───SabberStone
```

Run the following to compile SabberStone. Since we only need `SabberStoneCore.dll`, we can only compile this sub-project and ignore other projects in SabberStone.

```
cd SabberStone/SabberStoneCore
dotnet publish --configuration Release
```

Then, a `SabberStoneCore.dll` file should be written to `SabberStone/SabberStoneCore/bin/Release/netstandard2.0/SabberStoneCore.dll`, which will be used by EvoStone 2.1.

Next move to the `TestBed/DeckSearch` directory. From here you can run the `setup.py` script.

```
python3 setup.py
```
That's it! Your project is now setup to run experiments from the paper.

## Running Experiments (Locally)

The setup script created several empty folders in the `TestBed/DeckSearch` directory: `bin`, `active`, `boxes`, `logs`, and `train_log`. The `bin` folder holds the compiled executables. The `active` folder is used for initial communication between distributed nodes and for letting the workers know when the search is complete. The `boxes` folder is for sending neural networks to the `DeckEvaluator` and receiving results. The `logs` folder holds logging files for the neural net surrogate models and elite maps from the search. The `train_log` folder is the default place to log the information of the surrogate model while training surrogate model offline.

First we need to start the control node responsible for running our search (MAP-Elites, DSA-ME, etc). To do this, run the following command.

```
dotnet bin/DeckSearch.dll config/experiment/distrited_search/paladin_me_demo.tml
```

The first parameter passed is the config file for the experiment. Here we are running MAP-Elites to search for decks for hero `paladin`. However, the search isn't moving because it doesn't have any worker nodes to play games. To start a worker node, run the following command.

```
dotnet bin/DeckEvaluator.dll 1
```

This command starts a new DeckEvaluator node. The first parameter is the node ID. You can start multiple nodes locally, but you must specify a different node for each worker. The node will take a deck generated from the search algorithm and play specified number of games using that deck. Once the games are complete, the node will send results back to the control node and await a new deck.


## Running DeckSearch Experiment (Distributed using Slurm)

Run a python script to clean up left-over files from previous experiments, if any:
```
python setup_hpc.py
```

Then, run the following to schedule the experiment:
```
sh slurm/run_search_slurm.sh <config_file> <num_evaluators>
```
where `<config_file>` is the path to the configuration file and `<num_evaluators>` is the number of evaluators.


## Config Files

There are two types of config files for EvoStone 2.0. The first specifies the experiment level parameters (see below).

```
[Evaluation]
OpponentDeckSuite = "resources/decks/suites/starterMeta.tml"
DeckPools = ["resources/decks/pools/starterDecks.tml"]

[[Evaluation.PlayerStrategies]]
NumGames = 20
Strategy = "Control"

[Deckspace]
HeroClass = "paladin"
CardSets = ["CORE", "EXPERT1"]

[Search]
Category = "Distributed"
Type = "MAP-Elites"
ConfigFilename = "config/elite_map/paladin_me_demo_config.tml"
```

The config file specifies how many games are played, the opponents to play against, the algorithm to use, and other useful information.

Then, the `Search.ConfigFileName` param specifies the config file of the search algorithm (see below).

```
[Search]
InitialPopulation = 100
NumToEvaluate = 10000

[Map]
Type = "FixedFeature"
StartSize = 40
EndSize = 40

[[Map.Features]]
Name = "NumTurns"
MinValue = 5.0
MaxValue = 15.0

[[Map.Features]]
Name = "HandSize"
MinValue = 1.0
MaxValue = 7.0
```

This config file specifies the behavior dimensions for the map of elites and parameters specific to running the search algorithm.


## Experiments and Corresponding Config Files

For each algorithm in the paper, the correspoinding config file is the following:

| Algorithm Name | Config file |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| MAP-Elites                           | config/experiment/distrited_search/rogue_classic_miracle_me_w_elitedeck_nn_strategy.tml                                               |
| DSA-ME                               | config/experiment/surrogate_search/rogue_classic_miracle_surr_me_fcnn_w_elitedeck_nn_strategy_w_out_dist_test.tml                     |
| DSA-ME (without resetting)           | config/experiment/surrogate_search/rogue_classic_miracle_surr_me_fcnn_w_elitedeck_nn_strategy_w_out_dist_test_keep_surr_archive.tml   |
| DSA-ME (with Ancillary Data)         | config/experiment/surrogate_search/rogue_classic_miracle_surr_me_fcnn_w_elitedeck_nn_strategy.tml                                     |
| LSA-ME                               | config/experiment/surrogate_search/rogue_classic_miracle_surr_me_linear_w_elitedeck_nn_strategy.tml                                   |
| Offline DSA-ME                       | config/experiment/surrogate_search/rogue_classic_miracle_surr_me_fixed_fcnn_w_elitedeck_nn_strategy_default_target.tml                |
| Offline DSA-ME (with Ancillary Data) | config/experiment/surrogate_search/rogue_classic_miracle_surr_me_fixed_fcnn_w_elitedeck_nn_strategy.tml                               |
| Offline DSA-ME (Elite Data)          | config/experiment/surrogate_search/rogue_classic_miracle_surr_me_fixed_dsa-me-offline_fcnn_w_elitedeck_nn_strategy_default_target.tml |
| Offline DSA-ME (Surrogate Model)     | config/experiment/surrogate_search/rogue_classic_miracle_surr_me_fixed_dsa-me_fcnn_w_elitedeck_nn_strategy_default_target.tml         |
