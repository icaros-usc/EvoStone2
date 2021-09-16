# EvoStone

This project is contains the Hearthstone experiment of the paper *[Learning Emulation Models for Automated Hearthstone Deckbuilding]()*. The code base builds upon [EvoStone](https://github.com/tehqin/EvoStone), which contains Hearthstone experiments for the paper *[Covariance Matrix Adaptation for the Rapid Illumination of Behavior Space](https://arxiv.org/abs/1912.02400)*. The project contains distributed implementations of evolutionary algorithms EM-ME and all the correspoinding emulation models.

This project is designed to be run on a High-Performance Computing (HPC) cluster and is divided into two subprojects `DeckEvaluator` (for running Hearthstone games and collecting data from those games) and `DeckSearch` (for running distributed versions of each evolutionary algorithm). EvoStone is a unified .NET project and all subprojects can be compiled through a single command.

## Installation
To install the project, you need to install the [.NET Core 3.1](https://dotnet.microsoft.com/download) developer toolkit for your system. You may also need the [NuGet](https://docs.microsoft.com/en-us/nuget/install-nuget-client-tools) client tools for updating dependencies for the project. Included in this project is a `setup.py` script to make installation easier. Users can install Python 3 or follow the commands in the script for installation.

Next you need to compile [SabberStone](https://github.com/HearthSim/SabberStone), the Hearthstone simulator for running games. Follow the instructions on the SabberStone github compile the project into `SabberStone.dll`. Put `SabberStone` in the same root directory as `EvoStone`. Therefore the project structure should look like the follows:
```
project
|
└───EvoStone
|
└───SabberStone
```

Next move to the `TestBed/DeckSearch` directory. From here you can run the `setup.py` script.

```
python3 setup.py
```
That's it! Your project is now setup to run experiments from the paper.

## Running Experiments (Locally)

The setup script created three empty folders in the `TestBed/DeckSearch` directory: `bin`, `active`, `boxes`, and `logs`. The `active` folder is used for initial communication between distributed nodes and for letting the workers know when the search is complete. The `boxes` folder is for sending neural networks to the `DeckEvaluator` and receiving results. The `logs` folder holds CSV files for logging information about the neural net policies and elite maps from the search.

First we need to start the control node responsible for running our search (MAP-Elites, EM-ME, etc). To do this, run the following command.

```
dotnet bin/DeckSearch.dll config/experiment/distrited_search/paladin_me_demo.tml
```

The first parameter passed is the config file for the experiment. Here we are running MAP-Elites to search for decks for hero `paladin`. However, the search isn't moving because it doesn't have any worker nodes to play games. To start a worker node, run the following command.

```
dotnet bin/DeckEvaluator.dll 1
```

This command starts a new DeckEvaluator node. The first parameter is the node ID. You can start multiple nodes locally, but you must specify a different node for each worker. The node will take a deck generated from the search algorithm and play specified number of games using that deck. Once the games are complete, the node will send results back to the control node and await a new deck.


## Running DeckSearch Experiment (Distributed using Slurm)

Running the experiment on HPC using slurm is a bit more tricky because .NET is not supported natively there. Therefore, you need to first build a [Singularity](https://sylabs.io/docs/) container and use the provided script to run the experiment.

First, log into the HPC:
```
ssh <USC_ID>@discovery.usc.edu
```

Then, you have to build the Singularity container. USC HPC natively supports Singularity, so you can use it directly. Run the following to build the container:
```
cd TestBed/DeckSearch
sudo singularity build singularity/ubuntu_dotnet singularity/ubuntu_dotnet.def
```

You can also build the contrainer locally and copy it to the HPC using `scp`.

Then, run a python script to clean up left-over files from previous experiments, if any:
```
python setup_hpc.py
```

Finally, run the following to schedule the experiment:
```
sh slurm/run_slurm.sh <config_file> <num_evaluators>
```
where `<config_file>` is the path to the configuration file and `<num_evaluators>` is the number of evaluators.


## Config Files

There are two types of config files for EvoStone. The first specifies the experiment level parameters (see below).

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
