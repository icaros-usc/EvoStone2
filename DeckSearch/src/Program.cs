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
            // DataProcesser.PreprocessDeckDataWithOnehot();
            // var search = new DistributedSearch(args[0]);
            // search.Run();
        }
    }
}
