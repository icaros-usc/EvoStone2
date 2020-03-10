using DeckSearch.Search;
using System;
using System.IO;
using System.ComponentModel;
using System.Globalization;
using System.Collections.Generic;
using SabberStoneCore.Model;
using CsvHelper;

namespace DeckSearch
{
    class Program
    {
        public static void printCardsInfo()
        {
            var allCards = Cards.All;
            using (var writer = new StreamWriter("../../../surrogate-model/card_data.csv"))
            using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
            {    
                csv.WriteRecords(allCards);
            }

        }

        public static void readCards() {
            // read in data of all of the cards from SabberStone
            // used for surrogate model
            var allCards = Cards.All;
            int cnt = 0;
            foreach(var card in allCards) {
                foreach(PropertyDescriptor descriptor in TypeDescriptor.GetProperties(card)) {
                    Console.WriteLine(card.TargetingPredicate);
                    string name = descriptor.Name;
                    object value = descriptor.GetValue(card);
                    Console.WriteLine("{0}={1}",name,value);
                }
                Console.WriteLine();
                cnt+=1;
                if(cnt == 2) {
                    break;
                }
                
            }
        }
        static void Main(string[] args)
        {
            readCards();
            // printCardsInfo();
            // var search = new DistributedSearch(args[0]);
            // search.Run();
        }
    }
}
