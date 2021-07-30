using System;
using System.Linq;
using System.Threading;
using System.Diagnostics;
using System.Collections.Generic;

using SurrogateModel.Surrogate;
using DeckSearch.Search.MapElites;
using DeckSearch.Logging;

using SabberStoneUtil;
using SabberStoneUtil.Config;
using SabberStoneUtil.DataProcessing;
using SabberStoneUtil.Messaging;

namespace DeckSearch.Search
{
    /// <summary>
    /// Search that exploits SurrogateModel
    /// </summary>
    class SurrogatedSearch
    {
        /// <summary>
        /// Number of generations to run per round
        /// </summary>
        private int _numGeneration;

        /// <summary>
        /// Number of evaluations to run per generation
        /// </summary>
        private int _numToEvaluatePerGen;

        /// <summary>
        /// Log length of surrogate elite map
        /// </summary>
        private int _logLengthPerGen;

        /// <summary>
        /// Surrogate model
        /// </summary>
        private static SurrogateBaseModel _surrogateModel;

        /// <summary>
        /// Search Manager to communicate with DeckEvaluators
        /// </summary>
        private DeckSearchManager _searchManager;


        private RunningIndividualLog _individualLog;


        private Stopwatch _stopWatch = new Stopwatch();


        private int _numSurrogateEvals = 0;


        private int _numMAPElitesRun = 0;


        private string _surrogateElitesLogDir;


        private string[] _modelTargets;


        private bool _useFixedModel = false;


        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name = "configFilename">name of the configuation file</param>
        public SurrogatedSearch(string configFilename)
        {
            _searchManager = new DeckSearchManager(configFilename);
            var config = _searchManager.config;
            _numGeneration = config.Search.NumGeneration;
            _numToEvaluatePerGen = config.Search.NumToEvaluatePerGeneration;
            _numSurrogateEvals = _searchManager.searchAlgo.InitialPopulation();
            _logLengthPerGen = config.Search.LogLengthPerGen;

            // create surrogate elites log dir
            _surrogateElitesLogDir = System.IO.Path.Combine(
                _searchManager.log_dir_exp, "surrogate_elites");
            System.IO.Directory.CreateDirectory(_surrogateElitesLogDir);


            // configurate surrogate model
            if (config.Surrogate.Type == "DeepSetModel")
            {
                _surrogateModel = new DeepSetModel(
                    log_dir_exp: _searchManager.log_dir_exp,
                    model_targets: config.Surrogate.ModelTargets);
            }
            else if (config.Surrogate.Type == "FullyConnectedNN")
            {
                _surrogateModel = new FullyConnectedNN(
                    log_dir_exp: _searchManager.log_dir_exp,
                    model_targets: config.Surrogate.ModelTargets);
            }
            else if (config.Surrogate.Type == "LinearModel")
            {
                _surrogateModel = new LinearModel(
                    log_dir_exp: _searchManager.log_dir_exp,
                    model_targets: config.Surrogate.ModelTargets);
            }
            else if (config.Surrogate.Type == "FixedFCNN")
            {
                _useFixedModel = true;
                _surrogateModel = new FullyConnectedNN(
                    log_dir_exp: _searchManager.log_dir_exp,
                    model_targets: config.Surrogate.ModelTargets);
                _surrogateModel.LoadModel(config.Surrogate.FixedModelSavePath);
                Utilities.WriteLineWithTimestamp(
                    String.Format("Loaded model from {0}",
                                  config.Surrogate.ModelTargets));
            }
            else
            {
                throw new System.ArgumentException(
                    "Invalid model type: {0}", config.Surrogate.Type);
            }
            Utilities.WriteLineWithTimestamp(
                String.Format("{0} Surrogate model created.",
                              config.Surrogate.FixedModelSavePath));

            // get model targets
            _modelTargets = _surrogateModel.model_targets;

        }

        /// <summary>
        /// Helper function to convert Individual instance to LogIndividual instance that is used for SurrogateModel
        /// </summary>
        static private LogIndividual ConvertIndividual(Individual individual, bool includeStats = true)
        {
            LogIndividual logIndividual = new LogIndividual();
            logIndividual.Deck = string.Join("*", individual.GetCards());

            if(includeStats)
            {
                logIndividual.IndividualID = individual.ID;
                logIndividual.Parent = individual.ParentID;
                logIndividual.WinCount = (int)individual.OverallData.WinCount;
                logIndividual.AverageHealthDifference = individual.OverallData.AverageHealthDifference;
                logIndividual.DamageDone = individual.OverallData.DamageDone;
                logIndividual.NumTurns = individual.OverallData.NumTurns;
                logIndividual.CardsDrawn = individual.OverallData.CardsDrawn;
                logIndividual.HandSize = individual.OverallData.HandSize;
                logIndividual.ManaSpent = individual.OverallData.ManaSpent;
                logIndividual.ManaWasted = individual.OverallData.ManaWasted;
                logIndividual.StrategyAlignment = individual.OverallData.StrategyAlignment;
                logIndividual.Dust = (int)individual.OverallData.Dust;
                logIndividual.DeckManaSum = (int)individual.OverallData.DeckManaSum;
                logIndividual.DeckManaVariance = individual.OverallData.DeckManaVariance;
                logIndividual.NumMinionCards = (int)individual.OverallData.NumMinionCards;
                logIndividual.NumSpellCards = (int)individual.OverallData.NumSpellCards;
            }
            return logIndividual;
        }

        /// <summary>
        /// Helper function to convert a List of Individuals to a List of LogIndividuals
        /// </summary>
        static private List<LogIndividual> ConvertIndividuals(List<Individual> individuals, bool includeStats = true)
        {
            List<LogIndividual> logIndividuals = new List<LogIndividual>();
            foreach(var individual in individuals)
            {
                logIndividuals.Add(ConvertIndividual(individual, includeStats));
            }
            return logIndividuals;
        }

        /// <summary>
        /// Function to run SurrogateModel Evaluation and record results
        /// </summary>
        public void EvaluateOnSurrogate(List<Individual> individuals)
        {
            var logIndividuals = ConvertIndividuals(individuals, includeStats: false);
            var result = _surrogateModel.Predict(logIndividuals);

            // update statistics of individuals
            for(int i = 0; i < individuals.Count; i++)
            {
                var individual = individuals[i];
                individual.ID = this._numSurrogateEvals;
                this._numSurrogateEvals += 1;

                // put result into individual
                individual.OverallData = new OverallStatistics();
                for(int j = 0; j < _modelTargets.Length; j++)
                {
                    string target = _modelTargets[j];
                    individual.OverallData
                        .GetType()
                        .GetProperty(target)
                        .SetValue(individual.OverallData, result[i, j]);
                }

                // save fitness
                individual.Fitness = individual.OverallData.AverageHealthDifference;

                // set StrategyData as empty array to avoid exceptions
                individual.StrategyData = new StrategyStatistics[0];
            }
        }

        /// <summary>
        /// Helper function to convert data and do back prop
        /// </summary>
        private void BackProp(List<Individual> individuals)
        {
            if (individuals.Count == 0) {
                Utilities.WriteLineWithTimestamp(
                    "Buffer is empty. Skipping backprop...");
                return;
            }
            var logIndividuals = ConvertIndividuals(individuals);
            _surrogateModel.OnlineFit(logIndividuals);
        }

        private void GetElapsedTime()
        {
            // Get the elapsed time as a TimeSpan value.
            TimeSpan ts = _stopWatch.Elapsed;

            // Format and display the TimeSpan value.
            string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                ts.Hours, ts.Minutes, ts.Seconds,
                ts.Milliseconds / 10);
            Utilities.WriteLineWithTimestamp("RunTime " + elapsedTime);
        }

        /// <summary>
        /// Function to run the SurrogatedSearch algorithm
        /// </summary>
        public void Run()
        {
            // let the workers know that searchAlgo is avialble
            _searchManager.AnnounceWorkersStart();
            Utilities.WriteLineWithTimestamp("Begin Surrogated Search...");

            // generate initial population
            while(!_searchManager.searchAlgo.InitialPopulationEvaluated())
            {
                // dispatch jobs until the number reaches
                // initial population size
                _searchManager.FindNewWorkers();

                // dispatch initial population
                _searchManager.DispatchInitJobsToWorkers();

                // wait for workers to finish evaluating initial population
                _searchManager.FindDoneWorkers(
                    storeBuffer: true,
                    logFeatureMap: true);
                Thread.Sleep(1000);
            }

            _searchManager.numEvaledPerRun = 0;


            List<Individual> indsToTrain;

            while(_searchManager.searchAlgo.IsRunning())
            {
                // back prop using individuals in the buffer
                if (!_useFixedModel)
                {
                    indsToTrain = _searchManager._individualsBuffer.ToList();
                    indsToTrain.Shuffle();
                    BackProp(_searchManager._individualsBuffer.ToList());
                }

                // clear the surrogate map
                _searchManager.searchAlgo.ClearSurrogateMap();

                // run MAP-Elites on surrogate
                Utilities.WriteLineWithTimestamp(
                    String.Format("Running {0} generations of Map-Elites, each with {1} individuals",
                                  _numGeneration, _numToEvaluatePerGen));

                // verbose exactly 10 times
                int verboseLogLength = _numGeneration/10;

                for(int i=0; i<_numGeneration; i++)
                {
                    // generate one generation of individuals
                    List<Individual> currGeneration = new List<Individual>();
                    for(int j=0; j<_numToEvaluatePerGen; j++)
                    {
                        Individual choiceIndividual =
                            _searchManager.searchAlgo.GenerateIndividualFromSurrogateMap(
                                    CardReader._cardSet);
                        currGeneration.Add(choiceIndividual);
                    }
                    EvaluateOnSurrogate(currGeneration);

                    // log the evaluated individuals
                    // add evaluated individuals to feature map and outer feature map
                    foreach(var individual in currGeneration)
                    {
                        _searchManager.searchAlgo.AddToSurrogateFeatureMap(individual);
                    }

                    if ((i+1) % verboseLogLength == 0)
                    {
                        Utilities.WriteLineWithTimestamp(
                            String.Format("Generation {0} completed...", i+1));
                    }
                }

                // log feature map
                _searchManager.searchAlgo.LogSurrogateFeatureMap();

                // get elites to evaluate for real
                var elites = _searchManager.GetAllElitesFromSurrogateMap();

                // log all elites for the current MAP-Elites run
                string surrogate_log_file = System.IO.Path.Combine(_surrogateElitesLogDir,
                     String.Format("surrogate_elites_log{0}.csv",
                                   this._numMAPElitesRun));
                this._numMAPElitesRun += 1;

                this._individualLog = new RunningIndividualLog(surrogate_log_file);
                for (int i=0; i<elites.Count; i++)
                {
                    this._individualLog.LogIndividual(elites[i]);
                }

                // evaluate elites
                Utilities.WriteLineWithTimestamp(
                    String.Format("Get {0} elites. Start evaluation...",
                                  elites.Count));
                int eliteIdx = 0; // index of elite to dispatch, also the number of elites dispatched.
                while(_searchManager.numEvaledPerRun < elites.Count)
                {
                    _searchManager.FindNewWorkers();

                    // need to dispatch elites
                    if(eliteIdx < elites.Count)
                    {
                        eliteIdx += _searchManager.DispatchOneJobToWorker(
                            choiceIndividual: new Individual(elites[eliteIdx]));
                    }

                    // wait for workers to finish evaluating all elites
                    _searchManager.FindDoneWorkers(
                        storeBuffer: true,
                        keepIndID: true,
                        logFeatureMap: true);
                    _searchManager.FindOvertimeWorkers();
                    Thread.Sleep(1000);
                }

                // some verbose info
                Utilities.WriteLineWithTimestamp(
                    String.Format("Finished evaluating {0} elites",
                                  _searchManager.numEvaledPerRun));
                Utilities.WriteLineWithTimestamp(
                    String.Format("Current number of training individuals: {0}",
                                  _searchManager._individualsBuffer.Count));

                // reset run per run
                _searchManager.numEvaledPerRun = 0;
            }

            if (!_useFixedModel)
            {
                // Final back prop using individuals in the buffer
                indsToTrain = _searchManager._individualsBuffer.ToList();
                indsToTrain.Shuffle();
                BackProp(indsToTrain);
            }

            // Let the workers know that we are done.
            _searchManager.AnnounceWorkersDone();
        }
    }
}