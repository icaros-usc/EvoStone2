using DeckSearch.Search;
using DeckSearch.Surrogate;
using System;
using System.Collections.Generic;
using SabberStoneUtil.DataProcessing;

namespace DeckSearch
{
    class Program
    {
        static void Main(string[] args)
        {
            var model = new Model();
            model.Run();
            // DataProcessor.WriteDeckOnehotEncoding("../../../surrogate-model/encoding_deck_data.csv");
            // var search = new DistributedSearch(args[0]);
            // search.Run();
        }
    }
}
