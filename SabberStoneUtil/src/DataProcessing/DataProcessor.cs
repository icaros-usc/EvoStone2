using System;
using System.IO;
using System.Linq;
using System.ComponentModel;
using System.Globalization;
using System.Collections.Generic;
using SabberStoneCore.Model;
using SabberStoneCore.Enums;
using CsvHelper;
using CsvHelper.Configuration;

namespace SabberStoneUtil.DataProcessing
{
    public class DataProcessor
    {
        private static IEnumerable<Card> allCards => Cards.All;

        /// <summary>
        /// Basic(CORE) and classic(EXPERT1) cards
        /// </summary>
        private static IEnumerable<Card> initialCards = allCards.Where(c =>
                                                         (c.Set == CardSet.CORE || c.Set == CardSet.EXPERT1)
                                                       && c.Collectible
                                                       && c.Implemented
                                                       && c.Type != CardType.HERO
                                                       && c.Type != CardType.ENCHANTMENT
                                                       && c.Type != CardType.INVALID
                                                       && c.Type != CardType.HERO_POWER
                                                       && c.Type != CardType.TOKEN);
        private static void PrintCardInfo(Card card)
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
        /// Function to write processed deck feature to csv file
        /// </summary>
        private static void WriteNumericalDeckFeature(List<ProcessedIndividual> processedIndividuals)
        {
            // Write processed data to csv file
            Console.WriteLine("Writing processed data to csv...");
            using (var writer = new StreamWriter("../../../surrogate-model/processed_deck_data.csv"))
            using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
            {
                csv.WriteRecords(processedIndividuals);
            }
            Console.WriteLine("Data written to ../../../surrogate-model/processed_deck_data.csv");
        }


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
            WriteNumericalDeckFeature(processedIndividuals);
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
        public static void PrintDeckOnehotEncoding(String inPath)
        {
            (int [,] cardsEncoding, _) = PreprocessDeckDataWithOnehot(inPath);
            PrintDeckEncoding(cardsEncoding);
        }

        /// <summary>
        /// Write onehot encoded deck data to disk
        /// </summary>
        public static void WriteDeckOnehotEncoding(String inPath, String outPath)
        {
            (int[,] cardsEncoding, _) = PreprocessDeckDataWithOnehot(inPath);
            WriteDeckEncoding(cardsEncoding, outPath);
        }

        /// <summary>
        /// Generate (modified) one hot encoding feature
        /// </summary>
        public static (int[,], double[,]) PreprocessDeckDataWithOnehot(String path)
        {
            var logIndividualsList = readLogIndividuals(path).ToList();
            var initialCardsList = initialCards.ToList();

            // find a map from cards to index. This can save some computation time
            Dictionary<String, int> cardIndex = new Dictionary<string, int>();
            for(int i=0; i<initialCardsList.Count; i++)
            {
                String cardName = initialCardsList[i].Name;
                var findCardList = initialCards.Where(c => c.Name == cardName).ToList();
                if(findCardList.Count == 0) {
                    Console.WriteLine("Error: Could not find card <" +
                                      cardName +
                                      ">. Try to include more cardset and try again.");
                    return (null, null);
                }
                int index = initialCardsList.IndexOf(initialCards.Where(c => c.Name == cardName).ToList()[0]);
                cardIndex[cardName] = index;
            }

            // store encoding in a 2D array
            int [,] cardsEncoding = new int[logIndividualsList.Count, initialCardsList.Count]; // feature of surrogate model
            double [,] deckStats = new double[logIndividualsList.Count, 3]; // target of surrogate model

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
                deckStats[i,0] = logIndividualsList[i].AverageHealthDifference;
                deckStats[i,1] = logIndividualsList[i].NumTurns;
                deckStats[i,2] = logIndividualsList[i].HandSize;
            }
            return (cardsEncoding, deckStats);
        }
    }
}