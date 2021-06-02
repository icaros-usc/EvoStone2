using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Linq;
using System.Threading;

using Nett;

using DeckSearch.Search;

using SurrogateModel.Surrogate;

using SabberStoneUtil.DataProcessing;
using SabberStoneUtil.Config;
using SabberStoneUtil.Decks;
using SabberStoneUtil.Messaging;

using DeckEvaluator.Config;

namespace Analysis
{
    public class RemoveCardAnalysisManager : SearchManager
    {
        private List<LogIndividual> elitesToAnalyze;

        private string analysisLogDir;

        private const string REAL_SIM_DIR = "real_sim";

        private const string SURR_SIM_DIR = "surrogate_sim";

        private Queue<Individual> allIncompDeckInds;

        private int numToEval;

        private int numEvaled;

        private SurrogateBaseModel model;

        public static int CompareCellData(string s1, string s2)
        {
            string[] splitedData = s1.Split(":");
            double fitness1 = Convert.ToDouble(splitedData[5]);
            splitedData = s2.Split(":");
            double fitness2 = Convert.ToDouble(splitedData[5]);
            if (fitness1 == fitness2)
                return 0;
            else if (fitness1 < fitness2)
                return 1;
            else
                return -1;
        }

        public static string getModelPath(string expLogDir)
        {
            string modelDir = System.IO.Path.Combine(
                expLogDir,
                "surrogate_train_log",
                "surrogate_model"
            );
            int idx = 0;
            string modelSavePath = System.IO.Path.Combine(
                modelDir,
                String.Format("model{0}", idx)
            );
            while (System.IO.Directory.Exists(modelSavePath))
            {
                idx += 1;
                modelSavePath = System.IO.Path.Combine(
                    modelDir,
                    String.Format("model{0}", idx)
                );
            }

            modelSavePath = System.IO.Path.Combine(
                modelDir,
                String.Format("model{0}", idx - 1),
                "model.ckpt"
            );

            return modelSavePath;
        }

        public static void EvaluateOnSurrogate(
            List<string> incompleteDeck,
            SurrogateBaseModel model,
            string currSurrIndLogDir,
            string cardRemoved)
        {
            // evaluate on surrogate for comparasons
            var inCompLogIndividuals = new List<LogIndividual>();
            var inCompLogIndividual = new LogIndividual();
            inCompLogIndividual.Deck = String.Join("*", incompleteDeck);
            inCompLogIndividuals.Add(inCompLogIndividual);
            var result = model.Predict(inCompLogIndividuals);

            // store result
            var stats = new OverallStatistics();
            stats.AverageHealthDifference = result[0, 0];
            stats.NumTurns = result[0, 1];
            stats.HandSize = result[0, 2];

            // write result
            string gameLogPath = System.IO.Path.Combine(
                currSurrIndLogDir, String.Format("remove_card{0}.tml", cardRemoved));
            Toml.WriteFile<OverallStatistics>(stats, gameLogPath);
        }


        public RemoveCardAnalysisManager(
            string expLogDir,
            string configFilename
            ) : base(configFilename)
        {
            string expConfigPath = System.IO.Path.Combine(
                           expLogDir, "experiment_config.tml");
            string indsLogPath = System.IO.Path.Combine(
                expLogDir, "individual_log.csv");
            string elitesLogPath = System.IO.Path.Combine(
                expLogDir, "elite_map_log.csv");

            // create log directory for analysis
            analysisLogDir = System.IO.Path.Combine(
                expLogDir, "remove_card_analysis");
            if (System.IO.Directory.Exists(analysisLogDir))
            {
                System.IO.Directory.Delete(analysisLogDir, recursive: true);
            }

            // init card set
            var config = Toml.ReadFile<SabberStoneUtil.Config.Configuration>(expConfigPath);
            CardReader.Init(config);

            // load model
            string modelSavePath = getModelPath(expLogDir);

            // configurate surrogate model
            model = null;
            if (config.Surrogate.Type == "DeepSetModel")
            {
                model = new DeepSetModel();
            }
            else if (config.Surrogate.Type == "FullyConnectedNN")
            {
                model = new FullyConnectedNN();
            }
            model.LoadModel(modelSavePath);

            // read all logged individuals
            List<LogIndividual> inds = DataProcessor.readLogIndividuals(indsLogPath).ToList();

            // read elites
            List<string[]> rowData = DataProcessor.ReadElitesLogAsList(elitesLogPath);
            List<string> lastMap = rowData.Last().ToList();
            lastMap.RemoveAt(0);
            lastMap.Sort(CompareCellData); // ascending order
            elitesToAnalyze = new List<LogIndividual>();
            foreach (string cellData in lastMap)
            {
                string[] splitedData = cellData.Split(":");
                double fitness = Convert.ToDouble(splitedData[5]);
                int indID = Convert.ToInt32(splitedData[3]);
                // Console.WriteLine("{0}: {1}", indID, fitness);
                elitesToAnalyze.Add(inds.Where(ind => ind.IndividualID == indID).ToList().First());
            }

            // construct incomp decks
            Console.WriteLine("Generating incomplete decks and running them on surrogate model...");
            allIncompDeckInds = GenerateIncompleteDeckInds();
            numToEval = allIncompDeckInds.Count;
            numEvaled = 0;

            Console.WriteLine("Incomplete decks constructed.");
            Console.WriteLine("Evaluated all incomplete decks on surrogate model.");
        }


        private Queue<Individual> GenerateIncompleteDeckInds()
        {
            allIncompDeckInds = new Queue<Individual>();
            foreach (var elite in elitesToAnalyze)
            {
                // create directory for current elite
                string realSimDir = System.IO.Path.Combine(
                    analysisLogDir, REAL_SIM_DIR);
                string surrogateSimDir = System.IO.Path.Combine(
                    analysisLogDir, SURR_SIM_DIR);
                string currSimIndLogDir = System.IO.Path.Combine(
                    realSimDir,
                    String.Format("elite#{0}", elite.IndividualID));
                string currSurrIndLogDir = System.IO.Path.Combine(
                    surrogateSimDir,
                    String.Format("elite#{0}", elite.IndividualID));
                System.IO.Directory.CreateDirectory(currSimIndLogDir);
                System.IO.Directory.CreateDirectory(currSurrIndLogDir);

                // create and yield individuals with incomplete decks
                List<string> deck = elite.Deck.Split("*").ToList();
                HashSet<string> uniqueCards = new HashSet<string>(deck);
                foreach (string uniqueCard in uniqueCards)
                {
                    // create incomplete Deck that remove one card
                    List<string> incompDeck = new List<string>(deck);
                    incompDeck.RemoveAll(c => c == uniqueCard);

                    Individual incompDeckInd = new Individual(
                        incompDeck, CardReader._cardSet);
                    incompDeckInd.ParentID = elite.IndividualID;
                    incompDeckInd.CardRemoved = uniqueCard;
                    allIncompDeckInds.Enqueue(incompDeckInd);

                    // Evaluate current incomp deck on surrogate
                    EvaluateOnSurrogate(
                        incompDeck,
                        model,
                        currSurrIndLogDir,
                        uniqueCard);
                }
            }
            return allIncompDeckInds;
        }


        public void DispatchEvalJobsToWorkers()
        {
            while (_idleWorkers.Count > 0)
            {
                Individual choiceIndividual = allIncompDeckInds.Dequeue();
                if (DispatchOneJobToWorker(choiceIndividual) == 0)
                {
                    allIncompDeckInds.Enqueue(choiceIndividual);
                }
            }
        }


        public void FindDoneWorkers()
        {
            base.FindDoneWorkers((stableInd) =>
            {
                // Write the results
                var results = new ResultsMessage();
                results.PlayerDeck = new DeckParams();
                results.PlayerDeck.ClassName =
                    CardReader._heroClass.ToString();
                results.PlayerDeck.CardList = stableInd.GetCards();
                results.OverallStats = stableInd.OverallData;
                results.StrategyStats = stableInd.StrategyData;
                string currSimIndLogDir =
                    System.IO.Path.Combine(
                        analysisLogDir, REAL_SIM_DIR,
                        String.Format("elite#{0}", stableInd.ParentID));
                string gameLogPath =
                    System.IO.Path.Combine(
                        currSimIndLogDir,
                        String.Format("remove_card-{0}.tml",
                                      stableInd.CardRemoved));
                Toml.WriteFile<ResultsMessage>(results, gameLogPath);

                LogIndividual(stableInd, () =>
                {
                    numEvaled += 1;
                    Console.WriteLine("Eval ({0}/{1})", numEvaled, numToEval);
                    Console.WriteLine("Elite ID: {0}", stableInd.ParentID);
                    Console.WriteLine("Card Removed: {0}",
                                      stableInd.CardRemoved);
                });
            });
        }

        public bool Running() => allIncompDeckInds.Count > 0;
    }
}