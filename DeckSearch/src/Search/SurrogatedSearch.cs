using System;
using System.Threading;
using System.Diagnostics;
using System.Collections.Generic;

using DeckSearch.Config;
using SurrogateModel.Surrogate;
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
        /// Surrogate model
        /// </summary>
        private static Model _surrogateModel = new Model();

        /// <summary>
        /// Search Manager to communicate with DeckEvaluators
        /// </summary>
        private SearchManager _searchManager;

        private Stopwatch _stopWatch = new Stopwatch();

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name = "configFilename">name of the configuation file</param>
        public SurrogatedSearch(Configuration config, string configFilename)
        {
            _searchManager = new SearchManager(config, configFilename);
        }

        /// <summary>
        /// Helper function to convert Individual instance to LogIndividual instance that is used for SurrogateModel
        /// </summary>
        private LogIndividual ConvertIndividual(Individual individual, bool includeStats = true)
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
        private List<LogIndividual> ConvertIndividuals(List<Individual> individuals, bool includeStats = true)
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
        public void Evaluate(List<Individual> individuals)
        {
            var logIndividuals = ConvertIndividuals(individuals, includeStats: false);
            var (cardsEncoding, _) =
                DataProcessor.PreprocessDeckDataWithOnehotFromData(logIndividuals);
            var result = _surrogateModel.Predict(cardsEncoding);

            // update statistics of individuals
            foreach (var individual in individuals)
            {
                individual.OverallData = new OverallStatistics();
                individual.OverallData.AverageHealthDifference = result[0,0];
                individual.OverallData.NumTurns = result[0,1];
                individual.OverallData.HandSize = result[0,2];
            }
        }

        /// <summary>
        /// Helper function to convert data and do back prop
        /// </summary>
        private void BackProp(List<Individual> individuals)
        {
            var logIndividuals = ConvertIndividuals(individuals);
            var (cardsEncoding, deckStats) = DataProcessor.PreprocessDeckDataWithOnehotFromData(logIndividuals);
            _surrogateModel.OnlineFit(cardsEncoding, deckStats);
        }

        private void GetElapsedTime()
        {
            // Get the elapsed time as a TimeSpan value.
            TimeSpan ts = _stopWatch.Elapsed;

            // Format and display the TimeSpan value.
            string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                ts.Hours, ts.Minutes, ts.Seconds,
                ts.Milliseconds / 10);
            Console.WriteLine("RunTime " + elapsedTime);
        }

        /// <summary>
        /// Function to run the SurrogatedSearch algorithm
        /// </summary>
        public void Run()
        {
            // let the workers know that searchAlgo is avialble
            _searchManager.AnnounceWorkersStart();
            Console.WriteLine("Begin Surrogated Search...");

            // generate initial population
            while(!_searchManager._searchAlgo.InitialPopulationEvaluated())
            {
                // dispatch jobs until the number reaches initial population size
                if(!_searchManager._searchAlgo.InitialPopulationDispatched())
                {
                    _searchManager.FindNewWorkers();
                    _searchManager.DispatchJobsToWorkers();
                }

                // wait for workers to finish evaluating initial population
                _searchManager.FindDoneWorkers(storeBuffer: true, addToOuterFeatureMap: true);
                Thread.Sleep(1000);
            }

            while(_searchManager._searchAlgo.IsRunning())
            {
                // back prop using individuals in the buffer
                BackProp(_searchManager._individualsBuffer);
                _searchManager._individualsBuffer.Clear();

                // run certain number of iterations of map elites
                List<Individual> allIndividuals = new List<Individual>();
                int generationSize = 2000;
                int numIter = 160000/generationSize;
                Console.WriteLine("Running {0} generations of Map-Elites, each with {1} individuals",
                    numIter, generationSize);
                for(int i=0; i<numIter; i++)
                {
                    // generate one generation of individuals
                    List<Individual> currGeneration = new List<Individual>();
                    for(int j=0; j<generationSize; j++)
                    {
                        Individual choiceIndividual = _searchManager._searchAlgo.GenerateIndividual(_searchManager._cardSet);
                        currGeneration.Add(choiceIndividual);
                    }
                    Evaluate(currGeneration);

                    // add evaluated individuals to feature map and outer feature map
                    foreach(var individual in currGeneration)
                    {
                        _searchManager._searchAlgo.ReturnEvaluatedIndividual(individual);
                    }
                    Console.WriteLine("Generation {0} completed...", i);
                }

                // evaluate elites
                var elites = _searchManager.GetAllElites();
                Console.WriteLine("Get {0} elites. Start evaluation...", elites.Count);
                int eliteIdx = 0; // index of elite to dispatch, also the number of elites dispatched.
                while(_searchManager._individualsBuffer.Count < elites.Count)
                {
                    // need to dispatch elites
                    if(eliteIdx < elites.Count)
                    {
                        _searchManager.FindNewWorkers();
                        eliteIdx += _searchManager.DispatchOneJobToWorker(choiceIndividual: elites[eliteIdx]);
                    }

                    // wait for workers to finish evaluating all elites
                    _searchManager.FindDoneWorkers(storeBuffer: true, addToOuterFeatureMap: true);
                    Thread.Sleep(1000);
                }
                Console.WriteLine("Finished evaluating {0} elites", _searchManager._individualsBuffer.Count);
            }

            // Let the workers know that we are done.
            _searchManager.AnnounceWorkersDone();
        }
    }
}