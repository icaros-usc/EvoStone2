using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

using Nett;

using SabberStoneCore.Enums;
using SabberStoneCore.Model;

using SabberStoneUtil.Decks;
using SabberStoneUtil.Messaging;

using DeckSearch.Config;
using DeckSearch.Logging;
using DeckSearch.Search.MapElites;
using DeckSearch.Search.EvolutionStrategy;

namespace DeckSearch.Search
{
    /// <summary>
    /// A helper class to communicate with SurrogateModel and DeckEvaluator through File IO
    /// </summary>
    class SearchManager
    {
        public Queue<int> _runningWorkers;
        public Queue<int> _idleWorkers;
        public Dictionary<int, Individual> _individualStable;

        public string _configFilename { get; private set; }

        // Data to be used by Surrogate Model
        public List<Individual> _individualsBuffer = new List<Individual>();

        // Deck Info
        readonly public CardClass _heroClass;
        readonly public List<Card> _cardSet;

        // Search Algorithm
        public SearchAlgorithm _searchAlgo;

        // Logging
        private const string LOG_DIRECTORY = "logs/";
        private const string TRAIN_LOG_DIRECTORY = "train_log/";
        private const string INDIVIDUAL_LOG_FILENAME =
           LOG_DIRECTORY + "individual_log.csv";
        private const string CHAMPION_LOG_FILENAME =
           LOG_DIRECTORY + "champion_log.csv";
        private const string FITTEST_LOG_FILENAME =
           LOG_DIRECTORY + "fittest_log.csv";

        private const string TRAINING_DATA_LOG_FILENAME_PREFIX =
           TRAIN_LOG_DIRECTORY + "training_data";

        // record the index of the training data file
        private int training_idx = 0;

        private RunningIndividualLog _individualLog;
        private RunningIndividualLog _championLog;
        private RunningIndividualLog _fittestLog;

        private const string _boxesDirectory = "boxes/";
        public const string _inboxTemplate = _boxesDirectory
               + "deck-{0,4:D4}-inbox.tml";
        public const string _outboxTemplate = _boxesDirectory
               + "deck-{0,4:D4}-outbox.tml";

        // Let the workers know we are here.
        private const string _activeWorkerTemplate = _activeDirectory
               + "worker-{0,4:D4}.txt";
        private const string _activeDirectory = "active/";

        public const string _activeSearchPath = _activeDirectory
               + "search.txt";

        private double _maxWins;
        private double _maxFitness;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name = "configFilename">name of the configuation file</param>
        public SearchManager(string configFilename)
        {
            _maxWins = 0;
            _maxFitness = Int32.MinValue;
            _runningWorkers = new Queue<int>();
            _idleWorkers = new Queue<int>();
            _individualStable = new Dictionary<int, Individual>();

            // Grab the configuration info
            _configFilename = configFilename;
            var config = Toml.ReadFile<Configuration>(_configFilename);

            // Configuration for the search space
            _heroClass = CardReader.GetClassFromName(config.Deckspace.HeroClass);
            CardSet[] sets = CardReader.GetSetsFromNames(config.Deckspace.CardSets);
            _cardSet = CardReader.GetCards(_heroClass, sets);

            // Setup the logs to record the data on individuals
            InitLogs();

            // Set up search algorithm
            Console.WriteLine("Algo: " + config.Search.Type);
            if (config.Search.Type.Equals("MAP-Elites"))
            {
                var searchConfig = Toml.ReadFile<MapElitesParams>(config.Search.ConfigFilename);
                _searchAlgo = new MapElitesAlgorithm(searchConfig);
            }

            else if (config.Search.Type.Equals("EvolutionStrategy")) 
            {
                var searchConfig = Toml.ReadFile<EvolutionStrategyParams>(config.Search.ConfigFilename);
                _searchAlgo = new EvolutionStrategyAlgorithm(searchConfig);
            }
        }

        /// <summary>
        /// Helper function to initialize the logging related variables
        /// </summary>
        private void InitLogs()
        {
            _individualLog =
               new RunningIndividualLog(INDIVIDUAL_LOG_FILENAME);
            _championLog =
               new RunningIndividualLog(CHAMPION_LOG_FILENAME);
            _fittestLog =
               new RunningIndividualLog(FITTEST_LOG_FILENAME);
        }

        /// <summary>
        /// Helper function to write text to specified stream
        /// </summary>
        public static void WriteText(Stream fs, string s)
        {
            s += "\n";
            byte[] info = new UTF8Encoding(true).GetBytes(s);
            fs.Write(info, 0, info.Length);
        }

        /// <summary>
        /// Function to Find new instance of running DeckEvaluator
        /// </summary>
        public void FindNewWorkers()
        {
            // Look for new workers.
            string[] hailingFiles = Directory.GetFiles(_activeDirectory);
            foreach (string activeFile in hailingFiles)
            {
                string prefix = _activeDirectory + "worker-";
                if (activeFile.StartsWith(prefix))
                {
                    string suffix = ".txt";
                    int start = prefix.Length;
                    int end = activeFile.Length - suffix.Length;
                    string label = activeFile.Substring(start, end - start);
                    int workerId = Int32.Parse(label);
                    _idleWorkers.Enqueue(workerId);
                    _individualStable.Add(workerId, null);
                    File.Delete(activeFile);
                    Console.WriteLine("Found worker " + workerId);
                }
            }
        }

        /// <summary>
        /// Function to dispatch simulation job to DeckEvaluator through File IO
        /// </summary>
        public void DispatchJobToWorkers()
        {
            // Dispatch jobs to the available workers.
            while (_idleWorkers.Count > 0 && !_searchAlgo.IsBlocking())
            {
                int workerId = _idleWorkers.Dequeue();
                _runningWorkers.Enqueue(workerId);
                Console.WriteLine("Starting worker: " + workerId);

                Individual choiceIndividual = _searchAlgo.GenerateIndividual(_cardSet);

                string inboxPath = string.Format(SearchManager._inboxTemplate, workerId);
                SendWork(inboxPath, choiceIndividual);
                _individualStable[workerId] = choiceIndividual;
            }
        }


        /// <summary>
        /// Helper function to send simulation work to DeckEvaluator instance
        /// </summary>
        private void SendWork(string workerInboxPath, Individual cur)
        {
            var deckParams = new DeckParams();
            deckParams.ClassName = _heroClass.ToString().ToLower();
            deckParams.CardList = cur.GetCards();

            var msg = new PlayMatchesMessage();
            msg.Deck = deckParams;

            Toml.WriteFile<PlayMatchesMessage>(msg, workerInboxPath);
        }

        /// <summary>
        /// Function to find workers that are done with simulation and receieve the result
        /// </summary>
        public void FindDoneWorkers()
        {
            // Look for individuals that are done.
            int numActiveWorkers = _runningWorkers.Count;
            for (int i = 0; i < numActiveWorkers; i++)
            {
                int workerId = _runningWorkers.Dequeue();
                string inboxPath = string.Format(SearchManager._inboxTemplate, workerId);
                string outboxPath = string.Format(SearchManager._outboxTemplate, workerId);

                // Test if this worker is done.
                if (File.Exists(outboxPath) && !File.Exists(inboxPath))
                {
                    // Wait for the file to finish being written.
                    Console.WriteLine("Worker done: " + workerId);

                    ReceiveResults(outboxPath, _individualStable[workerId]);
                    _searchAlgo.ReturnEvaluatedIndividual(_individualStable[workerId]);
                    _individualsBuffer.Add(_individualStable[workerId]); // add evaluated individual to batch

                    Console.WriteLine("Batch Queue num: " + _individualsBuffer.Count);
                    // if reachs certain size, write stored training data to disk
                    if(_individualsBuffer.Count >= 128)
                    {
                        Console.WriteLine("Required batches completed, writing to disk for back prop");
                        LogTrainingIndividuals();
                    }
                    LogIndividual(_individualStable[workerId]);
                    _idleWorkers.Enqueue(workerId);
                }
                else
                {
                    _runningWorkers.Enqueue(workerId);
                }
            }
        }

        /// <summary>
        /// Heper function to receive results from DeckEvaluators
        /// </summary>
        private void ReceiveResults(string workerOutboxPath, Individual cur)
        {
            // Read the message and then delete the file.
            var results = Toml.ReadFile<ResultsMessage>(workerOutboxPath);
            File.Delete(workerOutboxPath);

            // Save the statistics for this individual.
            cur.OverallData = results.OverallStats;
            cur.StrategyData = results.StrategyStats;

            // Save which elements are relevant to the search
            cur.Fitness = cur.OverallData.AverageHealthDifference;

        }

        /// <summary>
        /// Function to log the info of a individual evaluated by DeckEvaluators
        /// </summary>
        public void LogIndividual(Individual cur)
        {
            var os = cur.OverallData;
            Console.WriteLine("------------------");
            Console.WriteLine(string.Format("Eval ({0}): {1}",
                              cur.ID,
                              string.Join("", cur.ToString())));
            Console.WriteLine("Win Count: " + os.WinCount);
            Console.WriteLine("Average Health Difference: "
                              + os.AverageHealthDifference);
            Console.WriteLine("Damage Done: " + os.DamageDone);
            Console.WriteLine("Num Turns: " + os.NumTurns);
            Console.WriteLine("Cards Drawn: " + os.CardsDrawn);
            Console.WriteLine("Hand Size: " + os.HandSize);
            Console.WriteLine("Mana Spent: " + os.ManaSpent);
            Console.WriteLine("Mana Wasted: " + os.ManaWasted);
            Console.WriteLine("Strategy Alignment: " + os.StrategyAlignment);
            Console.WriteLine("Dust: " + os.Dust);
            Console.WriteLine("Deck Mana Sum: " + os.DeckManaSum);
            Console.WriteLine("Deck Mana Variance: " + os.DeckManaVariance);
            Console.WriteLine("Num Minion Cards: " + os.NumMinionCards);
            Console.WriteLine("Num Spell Cards: " + os.NumSpellCards);
            Console.WriteLine("------------------");
            foreach (var fs in cur.StrategyData)
            {
                Console.WriteLine("WinCount: " + fs.WinCount);
                Console.WriteLine("Alignment: " + fs.Alignment);
                Console.WriteLine("------------------");
            }

            // Save stats
            bool didHitMaxWins =
               cur.OverallData.WinCount > _maxWins;
            bool didHitMaxFitness =
               cur.Fitness > _maxFitness;
            _maxWins = Math.Max(_maxWins, cur.OverallData.WinCount);
            _maxFitness = Math.Max(_maxFitness, cur.Fitness);

            // Log the individuals
            _individualLog.LogIndividual(cur);
            if (didHitMaxWins)
                _championLog.LogIndividual(cur);
            if (didHitMaxFitness)
                _fittestLog.LogIndividual(cur);
        }

        /// <summary>
        /// Function to log training individuals to Surrogate model
        /// </summary>
        public void LogTrainingIndividuals()
        {
            var _trainingDataLog =
                new RunningIndividualLog(TRAINING_DATA_LOG_FILENAME_PREFIX + training_idx.ToString() + ".csv");
            training_idx += 1;

            // Log training data to disk
            _trainingDataLog.LogIndividuals(_individualsBuffer);
            Console.WriteLine("Training data written to disk");

            // clear the buffer
            _individualsBuffer.Clear();
            Console.WriteLine("Buffer cleared");
        }
    }
}