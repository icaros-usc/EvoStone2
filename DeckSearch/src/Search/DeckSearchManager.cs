using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

using Nett;

using SabberStoneUtil.Config;

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
        /// List of individuals that are evaluated by the workers. The list is emptied periodically.
        /// Used by Surrogate search.
        /// </summary>
        public HashSet<Individual> _individualsBuffer { get; private set; }

        /// <summary>
        /// Search Algorithm
        /// </summary>
        public SearchAlgorithm searchAlgo { get; private set; }


        /// <summary>
        /// Number of individuals evaluated for each MAP-Elites run.
        /// This is used for Surrogated Search.
        /// </summary>
        public int numEvaledPerRun;


        private int _numToEvaluate = 0;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name = "configFilename">name of the configuation file</param>
        public DeckSearchManager(
            Configuration config,
            string configFilename
            ) : base(config, configFilename)
        {
            numEvaledPerRun = 0;
            _individualsBuffer = new HashSet<Individual>();

            // Set up search algorithm
            Console.WriteLine("Algo: " + config.Search.Type);
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
                    Console.WriteLine(
                        "Buffer Size: " + _individualsBuffer.Count);
                }
                LogIndividual(stableInd, () =>
                {
                    Console.WriteLine(string.Format("Eval ({0}/{1})",
                    searchAlgo.NumIndividualsEvaled(),
                    _numToEvaluate));
                });
            });


        }

        /// <summary>
        /// Helper function to determine wether current algorithm is Map-Elites
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