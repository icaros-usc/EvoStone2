using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Collections.Concurrent;
using Nett;

using SurrogateModel.Surrogate;

using SabberStoneUtil.DataProcessing;
using SabberStoneUtil.Config;
using SabberStoneUtil.Decks;
using SabberStoneUtil.Messaging;

using DeckSearch.Search;

using DeckEvaluator.Config;
using DeckEvaluator.Evaluation;

using SabberStoneCore.Enums;
using SabberStoneCore.Model;

using Tensorflow;
using NumSharp;
using static Tensorflow.Binding;

namespace Analysis
{
    class Program
    {
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
            List<LogIndividual> logIndividuals,
            SurrogateBaseModel model,
            string currSurrIndLogDir,
            int idx)
        {
            var result = model.Predict(logIndividuals);

            // store result
            var stats = new OverallStatistics();
            stats.AverageHealthDifference = result[0, 0];
            stats.NumTurns = result[0, 1];
            stats.HandSize = result[0, 2];

            // write result
            string gameLogPath = System.IO.Path.Combine(
                currSurrIndLogDir, String.Format("remove_card{0}.tml", idx));
            Toml.WriteFile<OverallStatistics>(stats, gameLogPath);
        }

        static void Main(string[] args)
        {
            // get exp log
            string expLogDir = args[0];

            string expConfigPath = System.IO.Path.Combine(
                expLogDir, "experiment_config.tml");
            string indsLogPath = System.IO.Path.Combine(
                expLogDir, "individual_log.csv");
            string elitesLogPath = System.IO.Path.Combine(
                expLogDir, "elite_map_log.csv");

            // create log directory for analysis
            string analysisLogDir = System.IO.Path.Combine(
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
            SurrogateBaseModel model = null;
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
            List<LogIndividual> elitesToAnalyze = new List<LogIndividual>();
            for (int i = 0; i < 10; i++)
            {
                var cellData = lastMap[i];
                string[] splitedData = cellData.Split(":");
                double fitness = Convert.ToDouble(splitedData[5]);
                int indID = Convert.ToInt32(splitedData[3]);
                // Console.WriteLine("{0}: {1}", indID, fitness);
                elitesToAnalyze.Add(inds.Where(ind => ind.IndividualID == indID).ToList().First());
            }

            // play game
            // set up opponent
            var deckConfig = Toml.ReadFile<DeckEvaluator.Config.Configuration>(
                args[1]);

            // Setup the pools of card decks for possible opponents.
            var deckPoolManager = new DeckPoolManager();
            deckPoolManager.AddDeckPools(deckConfig.Evaluation.DeckPools);

            // Setup test suites: (strategy, deck) combos to play against.
            var suiteConfig = Toml.ReadFile<DeckSuite>(
                deckConfig.Evaluation.OpponentDeckSuite);
            var gameSuite = new GameSuite(
                suiteConfig.Opponents, deckPoolManager);
            int numStrats = deckConfig.Evaluation.PlayerStrategies.Length;

            foreach (var logIndividual in elitesToAnalyze)
            {
                // create directory for current elite
                string realSimDir = System.IO.Path.Combine(
                    analysisLogDir, "real_sim");
                string surrogateSimDir = System.IO.Path.Combine(
                    analysisLogDir, "surrogate_sim");
                string currSimIndLogDir = System.IO.Path.Combine(
                    realSimDir,
                    String.Format("elite#{0}", logIndividual.IndividualID));
                string currSurrIndLogDir = System.IO.Path.Combine(
                    surrogateSimDir,
                    String.Format("elite#{0}", logIndividual.IndividualID));
                System.IO.Directory.CreateDirectory(currSimIndLogDir);
                System.IO.Directory.CreateDirectory(currSurrIndLogDir);

                // create player deck
                List<string> deck = logIndividual.Deck.Split("*").ToList();
                for (int j = 0; j < deck.Count(); j++)
                {
                    // create incompleteDeck that remove one card
                    List<string> incompleteDeck = new List<string>(deck);
                    incompleteDeck.RemoveAt(j);
                    Deck playerDeck = new Deck(
                        "paladin", incompleteDeck.ToArray());

                    // create stats
                    var stratStats = new StrategyStatistics[numStrats];
                    var overallStats = new OverallStatistics();
                    overallStats.UsageCounts = new int[playerDeck.CardList.Count];

                    // run the games
                    for (int i = 0; i < numStrats; i++)
                    {
                        // Setup the player with the current strategy
                        PlayerStrategyParams curStrat =
                            deckConfig.Evaluation.PlayerStrategies[i];
                        var player = new PlayerSetup(
                            playerDeck,
                            PlayerSetup.GetStrategy(curStrat.Strategy,
                                                    deckConfig.Network,
                                                    weights: null));

                        // set up opponents
                        List<PlayerSetup> opponents =
                            gameSuite.GetOpponents(curStrat.NumGames);

                        // launch the games
                        Console.WriteLine("Start games for elites #{0}", logIndividual.IndividualID);
                        var launcher = new GameDispatcher(player, opponents);

                        OverallStatistics stats = launcher.Run();
                        stratStats[i] = new StrategyStatistics();
                        stratStats[i].WinCount += stats.WinCount;
                        stratStats[i].Alignment += stats.StrategyAlignment;
                        overallStats.Accumulate(stats);
                    }

                    // Write the results
                    overallStats.ScaleByNumStrategies(numStrats);
                    var results = new ResultsMessage();
                    results.PlayerDeck = new DeckParams();
                    results.PlayerDeck.ClassName =
                        playerDeck.DeckClass.ToString();
                    results.PlayerDeck.CardList = incompleteDeck.ToArray();
                    results.OverallStats = overallStats;
                    results.StrategyStats = stratStats;
                    string gameLogPath =
                        System.IO.Path.Combine(
                            currSimIndLogDir,
                            String.Format("remove_card{0}.tml", j));
                    Toml.WriteFile<ResultsMessage>(results, gameLogPath);

                    // evaluate on surrogate for comparasons
                    var inCompLogIndividuals = new List<LogIndividual>();
                    var inCompLogIndividual = new LogIndividual();
                    inCompLogIndividual.Deck = String.Join("*", incompleteDeck);
                    inCompLogIndividuals.Add(inCompLogIndividual);
                    EvaluateOnSurrogate(
                        inCompLogIndividuals,
                        model,
                        currSurrIndLogDir, j);
                }
            }
        }
    }
}
