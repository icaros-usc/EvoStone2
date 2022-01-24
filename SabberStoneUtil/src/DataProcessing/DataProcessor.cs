using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.ComponentModel;
using System.Globalization;
using System.Collections.Generic;

using SabberStoneUtil.Config;

using SabberStoneCore.Model;
using SabberStoneCore.Enums;

using CsvHelper;
using CsvHelper.Configuration;

namespace SabberStoneUtil.DataProcessing
{
    public class DataProcessor
    {
        /// <summary>
        /// Number of cards in the search space
        /// </summary>
        public static int numCards = CardReader._cardSet.Count;

        private static IEnumerable<Card> allCards => Cards.All;

        /// <summary>
        // A map from cards to index. This can save some computation time
        /// </summary>
        private static Dictionary<String, int> cardIndex = new Dictionary<string, int>();

        /// <summary>
        /// static constructor
        /// </summary>
        static DataProcessor()
        {
            // construct card2index map
            for(int i=0; i<numCards; i++)
            {
                String cardName = CardReader._cardSet[i].Name;
                var findCardList = CardReader._cardSet.Where(c => c.Name == cardName).ToList();
                if(findCardList.Count == 0) {
                    Console.WriteLine("Error: Could not find card <" +
                                      cardName +
                                      ">. Try to include more cardset and try again.");
                    return;
                }
                int index = CardReader._cardSet.IndexOf(CardReader._cardSet.Where(c => c.Name == cardName).ToList()[0]);
                cardIndex[cardName] = index;
            }
        }

        // ****************** I/O Functions ******************
        public static void WriteCardIndex()
        {
            string jsonCardIndex = JsonSerializer.Serialize(cardIndex);
            Console.WriteLine(jsonCardIndex);
        }


        public static void PrintCardInfo(Card card)
        {
            foreach(PropertyDescriptor descriptor in TypeDescriptor.GetProperties(card))
            {
                Console.WriteLine(card.TargetingPredicate);
                string name = descriptor.Name;
                object value = descriptor.GetValue(card);
                Console.WriteLine("{0}={1}",name,value);
            }
            Console.WriteLine("----------------------------------------------------------------");
        }


        /// <summary>
        /// Helper Function to write processed list of objects to csv file
        /// </summary>
        private static void WriteToCsv<T>(List<T> objects, string path)
        {
            using (var writer = new StreamWriter(path))
            using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
            {
                csv.WriteRecords(objects);
            }
        }

        // /// <summary>
        // /// Function to write processed deck feature to csv file
        // /// </summary>
        // private static void WriteNumericalDeckFeature(List<ProcessedIndividual> processedIndividuals)
        // {
        //     // Write processed data to csv file
        //     Console.WriteLine("Writing processed data to csv...");
        //     using (var writer = new StreamWriter("../../../surrogate-model/processed_deck_data.csv"))
        //     using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
        //     {
        //         csv.WriteRecords(processedIndividuals);
        //     }
        //     Console.WriteLine("Data written to ../../../surrogate-model/processed_deck_data.csv");
        // }


        /// <summary>
        /// Function to read in logged individuals
        /// </summary>
        public static IEnumerable<LogIndividual> readLogIndividuals(String path)
        {
            var reader = new StreamReader(path);
            var csv = new CsvReader(reader, CultureInfo.InvariantCulture);
            IEnumerable<LogIndividual> logIndividuals = csv.GetRecords<LogIndividual>();
            return logIndividuals;
        }


        /// <summary>
        /// Print encoded deck data
        /// </summary>
        private static void PrintDeckEncoding(int [,] cardsEncoding)
        {
            for(int i=0; i<10; i++) {
                for(int j=0; j<cardsEncoding.GetLength(1); j++) {
                    Console.Write(cardsEncoding[i,j]);
                    Console.Write(" ");
                }
                Console.WriteLine();
            }
        }

        /// <summary>
        /// Write encoded deck data to disk
        /// </summary>
        private static void WriteDeckEncoding(int [,] cardsEncoding, String path)
        {
            using(StreamWriter writer = new StreamWriter(path))
            {
                for(int i=0; i<cardsEncoding.GetLength(0); i++)
                {
                    String line = "";
                    for(int j=0; j<cardsEncoding.GetLength(1); j++)
                    {
                        line += cardsEncoding[i,j].ToString();
                        if(j != cardsEncoding.GetLength(1)-1) {
                            line += ",";
                        }
                    }
                    writer.WriteLine(line);
                }
            }
        }

        /// <summary>
        /// Print onehot encoded deck data
        /// </summary>
        public static void PrintDeckOnehotEncoding(
            String inPath, string[] modelTargets)
        {
            (int [,] cardsEncoding, _) =
                PreprocessDeckOnehotFromFile(inPath, modelTargets);
            PrintDeckEncoding(cardsEncoding);
        }

        /// <summary>
        /// Write onehot encoded deck data to disk
        /// </summary>
        public static void WriteDeckOnehotEncoding(
            String inPath, String outPath, string[] modelTargets)
        {
            (int[,] cardsEncoding, _) =
                PreprocessDeckOnehotFromFile(inPath, modelTargets);
            WriteDeckEncoding(cardsEncoding, outPath);
        }

        public static List<string[]> ReadElitesLogAsList(String elitesLogPath)
        {
            List<string[]> rowData = new List<string[]>();
            var elitesLogFile = File.ReadAllLines(elitesLogPath);
            var elitesLogList = new List<string>(elitesLogFile);

            foreach (var rowMapData in elitesLogList)
            {
                var mapData = rowMapData.Split(',');
                rowData.Add(mapData);
            }
            return rowData;
        }

        // ***************** End I/O Functions *******************

        // ************ Preprocess Card/Deck Encoding from Data **************

        /// <summary>
        /// Generate numerical deck features
        /// </summary>
        public static void PreprocessDeckDataWithNumericalFeatures(String path)
        {
            Console.WriteLine("Processing raw deck data...");
            List<ProcessedIndividual> processedIndividuals;

            // logged individual, raw data set
            var logIndividuals = readLogIndividuals(path);

            // processed individual, processed data set
            processedIndividuals = new List<ProcessedIndividual>();
            foreach (var logIndividual in logIndividuals)
            {
                var ind = new ProcessedIndividual();
                ind.Index = logIndividual.IndividualID;

                string[] deckCards = logIndividual.Deck.Split('*');

                // calculate sum stats and calculate num related attributes
                foreach (var cardName in deckCards)
                {
                    var card = allCards.Where(c => c.Name == cardName && c.Collectible).First();

                    ind.DeckManaSum += card.Cost;
                    ind.AttackSum += card.ATK;
                    ind.HealthSum += card.Health;
                    ind.OverloadSum += card.Overload;

                    if (card.Type == CardType.MINION)
                    {
                        ind.NumMinionCards += 1;
                    }
                    if (card.Type == CardType.SPELL)
                    {
                        ind.NumSpellCards += 1;
                    }
                    if (card.Type == CardType.WEAPON)
                    {
                        ind.NumWeaponCards += 1;
                    }
                    if (card.Taunt)
                    {
                        ind.NumTaunt += 1;
                    }
                    if (card.Charge)
                    {
                        ind.NumCharge += 1;
                    }
                    if (card.Stealth)
                    {
                        ind.NumCharge += 1;
                    }
                    if (card.Poisonous)
                    {
                        ind.NumPoisonous += 1;
                    }
                    if (card.DivineShield)
                    {
                        ind.NumDivineShield += 1;
                    }
                    if (card.Windfury)
                    {
                        ind.NumWindfury += 1;
                    }
                    if (card.Rush)
                    {
                        ind.NumRush += 1;
                    }
                    if (card.ChooseOne)
                    {
                        ind.NumChooseOne += 1;
                    }
                    if (card.Combo)
                    {
                        ind.NumCombo += 1;
                    }
                    if (card.IsSecret)
                    {
                        ind.NumSecret += 1;
                    }
                    if (card.Deathrattle)
                    {
                        ind.NumDeathrattle += 1;
                    }
                    if (card.HasOverload)
                    {
                        ind.NumOverload += 1;
                    }
                }

                // calculate variance stats
                foreach (var cardName in deckCards)
                {
                    var card = allCards.Where(c => c.Name == cardName && c.Collectible).First();

                    ind.DeckManaVariance += Math.Pow(((double)card.Cost - ind.DeckManaSum/30.0), 2.0);
                    ind.AttackVariance += Math.Pow(((double)card.ATK - ind.AttackSum/30.0), 2.0);
                    ind.HealthVariance += Math.Pow(((double)card.Health - ind.HealthSum/30.0), 2.0);
                    ind.OverloadVariance += Math.Pow(((double)card.Overload - ind.OverloadSum/30.0), 2.0);
                }
                ind.DeckManaVariance /= 30;
                ind.AttackVariance /= 30;
                ind.HealthVariance /= 30;
                ind.OverloadVariance /= 30;

                // add processed individual to the list
                processedIndividuals.Add(ind);
            }

            Console.WriteLine("Processing Completed!");

            // write processed data
            WriteToCsv(processedIndividuals,
                       "../../../surrogate-model/processed_deck_data.csv");
        }


        /// <summary>
        /// Generate (modified) one hot encoding feature from data read from data
        /// </summary>
        public static (int[,], double[,]) PreprocessDeckOnehotFromData(List<LogIndividual> logIndividualsList, string[] modelTargets)
        {
            // store encoding in a 2D array
            int [,] cardsEncoding = new int[logIndividualsList.Count, numCards]; // feature of surrogate model
            for(int i=0; i<logIndividualsList.Count; i++)
            {
                string[] deckCards = logIndividualsList[i].Deck.Split('*');
                foreach (var cardName in deckCards)
                {
                    // find index of the card
                    int j = cardIndex[cardName];

                    // add encoding
                    cardsEncoding[i,j]++;
                }
            }
            double [,] deckStats = GetDeckStats(logIndividualsList, modelTargets); // target of surrogate model

            return (cardsEncoding, deckStats);
        }

        /// <summary>
        /// Generate encoding of a deck as a set of 30 one hot encoded cards from data
        /// </summary>
        public static (double [][][], double[,]) PreprocessCardsSetOnehotFromData(List<LogIndividual> logIndividualsList, string[] modelTargets)
        {
            /// store embedding in a 3D array
            // each deck contains 30 embedding vectors
            double[][][] deckEmbeddings =
                new double [logIndividualsList.Count][][];
            for(int i=0; i<logIndividualsList.Count; i++)
            {
                string[] deckCards = logIndividualsList[i].Deck.Split('*');
                double [][] deckEmbedding = new double[30][];
                for(int j=0; j<deckCards.Length; j++)
                {
                    // create onehot encoding of a card
                    double[] cardEmbedding = new double[numCards];
                    cardEmbedding[cardIndex[deckCards[j]]] = 1;
                    deckEmbedding[j] = cardEmbedding;
                }
                if (deckCards.Length < 30)
                {
                    for(int k=0; k<30-deckCards.Length; k++)
                    {
                        double[] cardEmbedding = new double[numCards];
                        deckEmbedding[deckCards.Length + k] = cardEmbedding;
                    }
                }

                deckEmbeddings[i] = deckEmbedding;
            }

            double[,] deckStats = GetDeckStats(
                logIndividualsList, modelTargets);
            return (deckEmbeddings, deckStats);
        }

        // ********* End Preprocess Card/Deck Encoding from Data ************

        //*********** Preprocess Card/Deck Encoding from File ****************

        /// <summary>
        /// Generate (modified) one hot encoding feature from data read from file
        /// </summary>
        /// <param name = "path">path of the file containing raw data</param>
        public static (int[,], double[,]) PreprocessDeckOnehotFromFile(String path, string[] modelTargets)
        {
            var logIndividualsList = readLogIndividuals(path).ToList();
            return PreprocessDeckOnehotFromData(logIndividualsList, modelTargets);
        }


        /// <summary>
        /// Generate encoding of a deck as a set of 30 one hot encoded cards from file
        /// </summary>
        public static (double [][][], double[,]) PreprocessCardsSetOnehotFromFile(string path, string[] modelTargets)
        {
            var logIndividualsList = readLogIndividuals(path).ToList();
            return PreprocessCardsSetOnehotFromData(
                logIndividualsList, modelTargets);
        }

        // ******** End Preprocess Card/Deck Encoding from File ************

        // ***************** Util Functions *********************

        /// <summary>
        /// Get Deck stats from list of LogIndividuals as the learning target
        /// </summary>
        public static double[,] GetDeckStats(
            List<LogIndividual> logIndsList,
            string[] modelTargets)
        {
            int numInds = logIndsList.Count;
            int numTargets = modelTargets.Length;
            // target of surrogate model
            double [,] deckStats = new double[numInds, numTargets];
            for(int i=0; i<numInds; i++)
            {
                for(int j=0; j<numTargets; j++)
                {
                    string target = modelTargets[j];
                    deckStats[i,j] = Convert.ToDouble(
                        logIndsList[i]
                        .GetType()
                        .GetProperty(target)
                        .GetValue(logIndsList[i]));
                    // convert win count to win rate
                    // NOTE:
                    // May need to change total number of game based on config.
                    if (target == "WinCount")
                    {
                        deckStats[i,j] /= 200.0;
                    }
                }
            }
            return deckStats;
        }

        /// <summary>
        /// Return true if the file is locked for the indicated access.
        /// </summary>
        private static bool FileIsLocked(string filename, FileAccess file_access)
        {
            // Try to open the file with the indicated access.
            try
            {
                FileStream fs =
                    new FileStream(filename, FileMode.Open, file_access);
                fs.Close();
                return false;
            }
            catch (IOException)
            {
                // file does not exist
                return true;
            }
            catch (Exception)
            {
                throw;
            }
        }

        // *************** End Util Functions ***********************
    }
}