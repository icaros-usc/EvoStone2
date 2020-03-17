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
    public class DataProcesser
    {
        private static IEnumerable<Card> allCards => Cards.All;
        public static void PrintCardInfo(Card card)
        {
            foreach(PropertyDescriptor descriptor in TypeDescriptor.GetProperties(card)) {
                Console.WriteLine(card.TargetingPredicate);
                string name = descriptor.Name;
                object value = descriptor.GetValue(card);
                Console.WriteLine("{0}={1}",name,value);
            }
            Console.WriteLine("----------------------------------------------------------------");
        }
        public static void WriteDeckData(List<ProcessedIndividual> processedIndividuals)
        {
            // Write processed data to csv file
            Console.WriteLine("Printing processed data to csv...");
            using (var writer = new StreamWriter("../../../surrogate-model/processed_deck_data.csv"))
            using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
            {
                csv.WriteRecords(processedIndividuals);
            }
            Console.WriteLine("Data written to ../../../surrogate-model/processed_deck_data.csv");
        }

        public static void PreprocessDeckData() {
            Console.WriteLine("Processing raw deck data...");
            List<ProcessedIndividual> processedIndividuals;

            using (var reader = new StreamReader("../../../surrogate-model/deck_search/individual_log.csv"))
            using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
            {
                // logged individual, raw data set
                var logIndividuals = csv.GetRecords<LogIndividual>();

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
            }
            Console.WriteLine("Processing Completed!");
            WriteDeckData(processedIndividuals);
        }
    }
}