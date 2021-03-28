using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

using Nett;

using SabberStoneCore.Enums;
using SabberStoneCore.Model;

using SabberStoneUtil.Decks;
using SabberStoneUtil.Messaging;
using SabberStoneUtil.Config;

using DeckSearch.Logging;
using DeckSearch.Search.MapElites;
using DeckSearch.Search.EvolutionStrategy;

namespace DeckSearch.Search
{
    /// <summary>
    /// A helper class to communicate with DeckEvaluator through File IO
    /// </summary>
    class SearchManager
    {
        /// <summary>
        /// A queue of running workers that are running deck evaluation
        /// </summary>
        public Queue<int> _runningWorkers;

        /// <summary>
        /// A queue of idle workers that could dispatch evaluation jobs to
        /// </summary>
        public Queue<int> _idleWorkers;

        /// <summary>
        /// A dict from ID of the workers to individuals that are evaluated by the workers
        public Dictionary<int, Individual> _individualStable;

        /// <summary>
        /// Filename of the configuation file
        /// </summary>
        public string _configFilename { get; private set; }

        /// <summary>
        /// List of individuals that are evaluated by the workers. The list is emptied periodically
        /// </summary>
        public List<Individual> _individualsBuffer = new List<Individual>();

        /// <summary>
        /// Search Algorithm
        /// </summary>
        public SearchAlgorithm _searchAlgo;

        // Logging objects
        private RunningIndividualLog _individualLog;
        private RunningIndividualLog _championLog;
        private RunningIndividualLog _fittestLog;

        /// <summary>
        /// Max number of win counts of all individuals
        /// </summary>
        private double _maxWins;

        /// <summary>
        /// Max fitness of all individuals
        /// </summary>
        private double _maxFitness;

        // Directory names
        private const string ACTIVE_DIRECTORY = "active/";
        private const string LOG_DIRECTORY = "logs/";
        private const string TRAIN_LOG_DIRECTORY = "train_log/";
        private const string BOXES_DIRECTORY = "boxes/";


        /// <summary>
        /// Prefix of the file path to write the trianing data for Surrogated Search
        /// </summary>
        private const string TRAINING_DATA_LOG_FILENAME_PREFIX =
           TRAIN_LOG_DIRECTORY + "training_data";

        /// <summary>
        /// File path for the Search Manager to send evaluation job to the evaluators
        /// </summary>
        public const string _inboxTemplate = BOXES_DIRECTORY
               + "deck-{0,4:D4}-inbox.tml";

        /// <summary>
        /// File path for the Search Manager to receive evaluation result from the evaluators
        /// </summary>
        public const string _outboxTemplate = BOXES_DIRECTORY
               + "deck-{0,4:D4}-outbox.tml";

        /// <summary>
        /// File path to write a file so that the workers know that a search is available
        /// </summary>
        public const string _activeSearchPath = ACTIVE_DIRECTORY
               + "search.txt";



        public string log_dir_exp { get; set; }

        /// <summary>
        // Record the index of the training data file.
        /// </summary>
        private int training_idx = 0;

        private int _numToEvaluate = 0;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name = "configFilename">name of the configuation file</param>
        public SearchManager(Configuration config, string configFilename)
        {
            _maxWins = 0;
            _maxFitness = Int32.MinValue;
            _runningWorkers = new Queue<int>();
            _idleWorkers = new Queue<int>();
            _individualStable = new Dictionary<int, Individual>();

            // Grab the configuration info
            _configFilename = configFilename;

            // set up log directory
            String log_dir_base = DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");
            String log_dir_exp = System.IO.Path.Combine(LOG_DIRECTORY, log_dir_base);
            this.log_dir_exp = log_dir_exp;
            System.IO.Directory.CreateDirectory(log_dir_exp);

            // write the config file to the log directory for future reference
            string config_out_path = System.IO.Path.Combine(log_dir_exp, "experiment_config.tml");
            Toml.WriteFile<Configuration>(config, config_out_path);

            // Setup the logs to record the data on individuals
            InitLogs(log_dir_exp);

            // Set up search algorithm
            Console.WriteLine("Algo: " + config.Search.Type);
            if (config.Search.Type.Equals("MAP-Elites"))
            {
                var searchConfig = Toml.ReadFile<MapElitesParams>(config.Search.ConfigFilename);
                _numToEvaluate = searchConfig.Search.NumToEvaluate;
                _searchAlgo = new MapElitesAlgorithm(searchConfig, log_dir_exp);
            }

            else if (config.Search.Type.Equals("EvolutionStrategy"))
            {
                var searchConfig = Toml.ReadFile<EvolutionStrategyParams>(config.Search.ConfigFilename);
                _numToEvaluate = searchConfig.Search.NumToEvaluate;
                _searchAlgo = new EvolutionStrategyAlgorithm(searchConfig);
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
        /// Helper function to write text to specified stream
        /// </summary>
        public static void WriteText(Stream fs, string s)
        {
            s += "\n";
            byte[] info = new UTF8Encoding(true).GetBytes(s);
            fs.Write(info, 0, info.Length);
        }

        /// <summary>
        /// Function to announce DeckEvaluator instances that SearchAlgorithm is available
        /// <summary>
        public void AnnounceWorkersStart()
        {
            using (FileStream ow = File.Open(_activeSearchPath,
                     FileMode.Create, FileAccess.Write, FileShare.None))
            {
                WriteText(ow, "MAP Elites");
                WriteText(ow, _configFilename);
                ow.Close();
            }
        }

        public void AnnounceWorkersDone()
        {
            File.Delete(_activeSearchPath);
        }

        /// <summary>
        /// Function to Find new instance of running DeckEvaluator
        /// </summary>
        public void FindNewWorkers()
        {
            // Look for new workers.
            string[] hailingFiles = Directory.GetFiles(ACTIVE_DIRECTORY);
            foreach (string activeFile in hailingFiles)
            {
                string prefix = ACTIVE_DIRECTORY + "worker-";
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
        /// Function to dispatch multiple simulation jobs to DeckEvaluator through File IO
        /// </summary>
        public void DispatchJobsToWorkers()
        {
            // Dispatch jobs to the available workers.
            while (_idleWorkers.Count > 0 && !_searchAlgo.IsBlocking())
            {
                Individual choiceIndividual = _searchAlgo.GenerateIndividual(CardReader._cardSet);
                DispatchOneJobToWorker(choiceIndividual);
            }
        }

        /// <summary>
        /// Function to dispatch one simulation job to DeckEvaluator
        /// </summary>
        /// <returns> 1 for success, 0 for failure.
        public int DispatchOneJobToWorker(Individual choiceIndividual)
        {
            if (_idleWorkers.Count == 0)
            {
                return 0;
            }
            int workerId = _idleWorkers.Dequeue();
            _runningWorkers.Enqueue(workerId);
            Console.WriteLine("Starting worker: " + workerId);
            string inboxPath = string.Format(SearchManager._inboxTemplate, workerId);
            SendWork(inboxPath, choiceIndividual);
            _individualStable[workerId] = choiceIndividual;
            return 1;
        }


        /// <summary>
        /// Helper function to send simulation work to DeckEvaluator instance
        /// </summary>
        private void SendWork(string workerInboxPath, Individual cur)
        {
            var deckParams = new DeckParams();
            deckParams.ClassName = CardReader._heroClass.ToString().ToLower();
            deckParams.CardList = cur.GetCards();

            var msg = new PlayMatchesMessage();
            msg.Deck = deckParams;

            Toml.WriteFile<PlayMatchesMessage>(msg, workerInboxPath);
        }

        /// <summary>
        /// Function to find DeckEvaluator instances that are done with simulation and receieve the result
        /// </summary>
        public void FindDoneWorkers(bool storeBuffer = false, bool keepIndID = false)
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
                    int originalID = _individualStable[workerId].ID;
                    _searchAlgo.AddToFeatureMap(_individualStable[workerId]);
                    _searchAlgo.LogFeatureMap();
                    // For Surrogate Search we need to keep the ID.
                    if (keepIndID)
                    {
                        _individualStable[workerId].ID = originalID;
                    }

                    // store done individual to a tmp buffer
                    if (storeBuffer)
                    {
                        _individualsBuffer.Add(_individualStable[workerId]); // add evaluated individual to batch
                        Console.WriteLine("Buffer Size: " + _individualsBuffer.Count);
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
            Console.WriteLine(string.Format("Eval ({0}/{1}): {2}",
                              _searchAlgo.NumIndividualsEvaled(),
                              _numToEvaluate,
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
                Console.WriteLine("------------------\n");
            }

            Console.WriteLine("\n");

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
        /// (deprecated) Function to log training individuals to Surrogate model
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

        /// <summary>
        /// Helper function to determine wether current algorithm is Map-Elites
        private bool IsMapElitesAlgo()
        {
            if (_searchAlgo.GetType().Equals(typeof(MapElitesAlgorithm)))
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

        public List<Individual> GetAllElitesFromSurrogateMap(int num=-1)
        {
            if (!IsMapElitesAlgo())
            {
                Console.WriteLine("Warning: {0} does not have elites!", _searchAlgo.GetType());
                return null;
            }

            var elites = _searchAlgo.GetAllElitesFromSurrogateMap();

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