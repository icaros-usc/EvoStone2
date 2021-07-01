using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

using Nett;

using SabberStoneUtil;
using SabberStoneUtil.Decks;
using SabberStoneUtil.Messaging;
using SabberStoneUtil.Config;

using DeckSearch.Logging;

namespace DeckSearch.Search
{
    /// <summary>
    /// A helper class to communicate with DeckEvaluator through File IO
    /// </summary>
    public class SearchManager
    {
        /// <summary>
        /// A queue of running workers that are running deck evaluation
        /// </summary>
        protected Queue<int> _runningWorkers { get; private set; }

        /// <summary>
        /// A queue of idle workers that could dispatch evaluation jobs to
        /// </summary>
        protected Queue<int> _idleWorkers { get; private set; }

        /// <summary>
        /// A dict from ID of the workers to individuals that are evaluated by the workers
        /// </summary>
        protected Dictionary<int, Individual> _individualStable { get; private set; }


        /// <summary>
        /// A dict from ID of the workers to start time of the worker job.
        /// </summary>
        protected Dictionary<int, DateTime> _workerRunningTimes { get; private set; }


        /// <summary>
        /// Filename of the configuation file
        /// </summary>
        public string _configFilename { get; private set; }


        /// <summary>
        /// Config of the search.
        /// </summary>
        public Configuration config { get; private set; }


        // Directory names
        private const string ACTIVE_DIRECTORY = "active/";
        private const string BOXES_DIRECTORY = "boxes/";


        /// <summary>
        /// File path for the Search Manager to send evaluation job to the evaluators
        /// </summary>
        protected const string _inboxTemplate = BOXES_DIRECTORY
               + "deck-{0,4:D4}-inbox.tml";

        /// <summary>
        /// File path for the Search Manager to receive evaluation result from the evaluators
        /// </summary>
        protected const string _outboxTemplate = BOXES_DIRECTORY
               + "deck-{0,4:D4}-outbox.tml";

        /// <summary>
        /// File path to write a file so that the workers know that a search is available
        /// </summary>
        protected const string _activeSearchPath = ACTIVE_DIRECTORY
               + "search.txt";




        // private int _numToEvaluate = 0;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name = "configFilename">name of the configuation file</param>
        public SearchManager(string configFilename)
        {
            _runningWorkers = new Queue<int>();
            _idleWorkers = new Queue<int>();
            _individualStable = new Dictionary<int, Individual>();
            _workerRunningTimes = new Dictionary<int, DateTime>();

            // Grab the configuration info
            _configFilename = configFilename;
            config = Toml.ReadFile<Configuration>(_configFilename);
        }



        /// <summary>
        /// Helper function to write text to specified stream
        /// </summary>
        private static void WriteText(Stream fs, string s)
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
            string[] hailingFiles = new string[] { };
            try
            {
                hailingFiles = Directory.GetFiles(ACTIVE_DIRECTORY);
            }
            catch (System.IO.IOException e)
            {
                Console.WriteLine("IOException catched while reading hailing files. Will retry...");
                Console.WriteLine("###########");
                Console.WriteLine(e.StackTrace);
                Console.WriteLine("###########");
                return;
            }
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
                    // avoid key error
                    if (!_individualStable.ContainsKey(workerId))
                    {
                        _individualStable.Add(workerId, null);
                        Utilities.WriteLineWithTimestamp(
                            String.Format("Found worker: {0} ", workerId));
                    }
                    try
                    {
                        File.Delete(activeFile);
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine("Exception while deleting: {0}",
                                          e.GetType().ToString());
                        Console.WriteLine(e.StackTrace);
                    }
                }
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
            string inboxPath = string.Format(_inboxTemplate, workerId);
            SendWork(inboxPath, choiceIndividual);
            _workerRunningTimes[workerId] = DateTime.UtcNow;
            Utilities.WriteLineWithTimestamp(
                String.Format("Worker start: {0}", workerId));

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
        public void FindDoneWorkers(System.Action<Individual> ProcessResult)
        {
            // Look for individuals that are done.
            int numActiveWorkers = _runningWorkers.Count;
            for (int i = 0; i < numActiveWorkers; i++)
            {
                int workerId = _runningWorkers.Dequeue();
                string inboxPath = string.Format(DeckSearchManager._inboxTemplate, workerId);
                string outboxPath = string.Format(DeckSearchManager._outboxTemplate, workerId);

                // Test if this worker is done.
                if (File.Exists(outboxPath) && !File.Exists(inboxPath))
                {
                    // Wait for the file to finish being written.
                    Utilities.WriteLineWithTimestamp(
                        String.Format("Worker done: {0}", workerId));

                    ReceiveResults(outboxPath, _individualStable[workerId]);
                    ProcessResult(_individualStable[workerId]);

                    _idleWorkers.Enqueue(workerId);
                    _workerRunningTimes.Remove(workerId);
                }
                else
                {
                    _runningWorkers.Enqueue(workerId);
                }
            }
        }

        /// <summary>
        /// If a worker does not return the job in 15 min, worker maybe dead.
        /// Resent the job to another worker.
        /// </summary>
        public void FindOvertimeWorkers()
        {
            var _workerRunningTimesCopy =
                new Dictionary<int, DateTime>(_workerRunningTimes);
            foreach (var item in _workerRunningTimesCopy)
            {
                int workerId = item.Key;
                TimeSpan timeDiff = DateTime.UtcNow - item.Value;
                int timeDiffMillisec = Convert.ToInt32(
                    timeDiff.TotalMilliseconds);
                if (timeDiffMillisec >= 15 * 60 * 1000)
                {
                    Utilities.WriteLineWithTimestamp(
                        String.Format("Worker {0} might be dead. Redispatching the job...", workerId));
                    // attempt to resend the job
                    // may fail for no idle workers
                    if (Convert.ToBoolean(
                        DispatchOneJobToWorker(_individualStable[workerId])))
                    {
                        Utilities.WriteLineWithTimestamp(
                            String.Format("Redispatching worker {0} succeeded.", workerId));
                        _workerRunningTimes.Remove(workerId);
                    }
                    else
                    {
                        Utilities.WriteLineWithTimestamp(
                            String.Format("Redispatching worker {0} failed. Will retry.", workerId));
                    }
                }
            }
        }

        /// <summary>
        /// Heper function to receive results from DeckEvaluators
        /// </summary>
        protected void ReceiveResults(string workerOutboxPath, Individual cur)
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
        protected void LogIndividual(
            Individual cur,
            System.Action LogProgress)
        {
            var os = cur.OverallData;
            Console.WriteLine("------------------");
            LogProgress();
            Console.WriteLine("Solution Deck: " + cur.ToString());
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
        }
    }
}