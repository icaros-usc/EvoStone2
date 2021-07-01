using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

using Nett;

using SabberStoneUtil;
using SabberStoneUtil.Config;

using DeckSearch.Logging;
using DeckSearch.Search.MapElites;
using DeckSearch.Search.EvolutionStrategy;
using DeckSearch.Search.RandomSearch;

namespace DeckSearch.Search
{
    /// <summary>
    /// A helper class to communicate with DeckEvaluator through File IO
    /// </summary>
    class DeckSearchManager : SearchManager
    {
        /// <summary>
        /// Number of individuals evaluated for each MAP-Elites run.
        /// This is used for Surrogated Search.
        /// </summary>
        public int numEvaledPerRun;

        /// <summary>
        /// List of individuals that are evaluated by the workers. The list is emptied periodically.
        /// Used by Surrogate search.
        /// </summary>
        public HashSet<Individual> _individualsBuffer { get; private set; }

        /// <summary>
        /// Search Algorithm
        /// </summary>
        public SearchAlgorithm searchAlgo { get; private set; }

        /// <summary>
        /// Log directory.
        /// </summary>
        public string log_dir_exp { get; private set; }

        // Logging objects
        private RunningIndividualLog _individualLog;
        private RunningIndividualLog _championLog;
        private RunningIndividualLog _fittestLog;

        /// <summary>
        /// Total number of evaluations
        /// </summary>
        private int _numToEvaluate = 0;

        /// <summary>
        /// Root log directory
        /// </summary>
        private const string LOG_DIRECTORY = "logs/";

        /// <summary>
        /// Max number of win counts of all individuals
        /// </summary>
        private double _maxWins;

        /// <summary>
        /// Max fitness of all individuals
        /// </summary>
        private double _maxFitness;


        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name = "configFilename">name of the configuation file</param>
        public DeckSearchManager(
            string configFilename
            ) : base(configFilename)
        {
            numEvaledPerRun = 0;
            _maxWins = 0;
            _maxFitness = Int32.MinValue;
            _individualsBuffer = new HashSet<Individual>();

            // set up log directory
            String log_dir_base = DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");
            log_dir_base += "_" + config.Search.Category +
                            "_" + config.Search.Type;
            if (config.Surrogate != null)
            {
                log_dir_base += "_" + config.Surrogate.Type;
            }
            String log_dir_exp = System.IO.Path.Combine(LOG_DIRECTORY, log_dir_base);
            this.log_dir_exp = log_dir_exp;
            System.IO.Directory.CreateDirectory(log_dir_exp);

            // write the config file to the log directory for future reference
            string config_out_path = System.IO.Path.Combine(log_dir_exp, "experiment_config.tml");
            Toml.WriteFile<Configuration>(config, config_out_path);

            // Setup the logs to record the data on individuals
            InitLogs(log_dir_exp);

            // Set up search algorithm
            Utilities.WriteLineWithTimestamp(
                "Search Algorithm: " + config.Search.Type);
            if (config.Search.Type.Equals("MAP-Elites"))
            {
                var searchConfig = Toml.ReadFile<MapElitesParams>(
                    config.Search.ConfigFilename);
                _numToEvaluate = searchConfig.Search.NumToEvaluate;
                searchAlgo = new MapElitesAlgorithm(searchConfig, log_dir_exp);
            }

            else if (config.Search.Type.Equals("EvolutionStrategy"))
            {
                var searchConfig = Toml.ReadFile<EvolutionStrategyParams>(config.Search.ConfigFilename);
                _numToEvaluate = searchConfig.Search.NumToEvaluate;
                searchAlgo = new EvolutionStrategyAlgorithm(searchConfig);
            }

            else if (config.Search.Type.Equals("RandomSearch"))
            {
                var searchConfig = Toml.ReadFile<RandomSearchParams>(
                    config.Search.ConfigFilename);
                _numToEvaluate = searchConfig.Search.NumToEvaluate;
                searchAlgo = new RandomSearchAlgorithm(
                    searchConfig, log_dir_exp);
            }
        }


        /// <summary>
        /// Helper function to initialize the logging related objects
        /// </summary>
        private void InitLogs(string log_dir_exp)
        {
            // File path to log all individuals
            string INDIVIDUAL_LOG_FILENAME = System.IO.Path.Combine(log_dir_exp, "individual_log.csv");

            // File path to log the champion individuals
            string CHAMPION_LOG_FILENAME = System.IO.Path.Combine(log_dir_exp, "champion_log.csv");

            // File path to log the fittest individuals
            string FITTEST_LOG_FILENAME = System.IO.Path.Combine(log_dir_exp, "fittest_log.csv");

            _individualLog =
               new RunningIndividualLog(INDIVIDUAL_LOG_FILENAME);
            _championLog =
               new RunningIndividualLog(CHAMPION_LOG_FILENAME);
            _fittestLog =
               new RunningIndividualLog(FITTEST_LOG_FILENAME);
        }


        /// <summary>
        /// Function to dispatch multiple simulation jobs to DeckEvaluator through File IO
        /// </summary>
        public void DispatchSearchJobsToWorkers()
        {
            // Dispatch jobs to the available workers.
            while (_idleWorkers.Count > 0 && !searchAlgo.IsBlocking())
            {
                Individual choiceIndividual = searchAlgo.GenerateIndividual(CardReader._cardSet);
                DispatchOneJobToWorker(choiceIndividual);
            }
        }


        /// <summary>
        /// Function to find DeckEvaluator instances that are done with simulation and receieve the result
        /// </summary>
        public void FindDoneWorkers(
            bool storeBuffer = false,
            bool keepIndID = false,
            bool logFeatureMap = true)
        {
            base.FindDoneWorkers((stableInd) =>
            {
                int originalID = stableInd.ID;
                searchAlgo.AddToFeatureMap(stableInd);

                if (logFeatureMap)
                {
                    searchAlgo.LogFeatureMap();
                }

                // A new ID would be given while the individual is added to
                // the feature map.
                // For Surrogate Search we need to keep the ID.
                if (keepIndID)
                {
                    stableInd.ID = originalID;
                }

                // store done individual to a tmp buffer
                if (storeBuffer)
                {
                    _individualsBuffer.Add(stableInd); // add evaluated individual to batch
                    numEvaledPerRun += 1;
                    Utilities.WriteLineWithTimestamp(
                        "Buffer Size: " + _individualsBuffer.Count);
                }
                LogIndividual(stableInd, () =>
                {
                    Console.WriteLine(string.Format("Eval ({0}/{1})",
                    searchAlgo.NumIndividualsEvaled(),
                    _numToEvaluate));
                });

                // Save stats
                bool didHitMaxWins = stableInd.OverallData.WinCount > _maxWins;
                bool didHitMaxFitness = stableInd.Fitness > _maxFitness;
                _maxWins = Math.Max(_maxWins, stableInd.OverallData.WinCount);
                _maxFitness = Math.Max(_maxFitness, stableInd.Fitness);

                // Log the individuals
                _individualLog.LogIndividual(stableInd);
                if (didHitMaxWins)
                    _championLog.LogIndividual(stableInd);
                if (didHitMaxFitness)
                    _fittestLog.LogIndividual(stableInd);
            });
        }


        /// <summary>
        /// Helper function to determine wether current algorithm is Map-Elites.
        /// </summary>
        private bool IsMapElitesAlgo()
        {
            if (searchAlgo.GetType().Equals(typeof(MapElitesAlgorithm)))
            {
                return true;
            }
            return false;
        }


        /// <summary>
        /// For Map-Elite algorithm, get all elites from the FeatureMap
        /// </summary>
        /// <param name="num">Number of elites to choose. Return all if -1.
        /// </param>
        public List<Individual> GetAllElitesFromSurrogateMap(int num = -1)
        {
            if (!IsMapElitesAlgo())
            {
                Console.WriteLine("Warning: {0} does not have elites!", searchAlgo.GetType());
                return null;
            }

            var elites = searchAlgo.GetAllElitesFromSurrogateMap();

            if (num == -1)
            {
                return elites;
            }

            var random = new Random();
            List<Individual> choiceElites = new List<Individual>();
            while (choiceElites.Count < num)
            {
                choiceElites.Add(elites[random.Next(elites.Count)]);
            }
            return choiceElites;
        }
    }
}