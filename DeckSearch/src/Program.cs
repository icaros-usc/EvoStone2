﻿using DeckSearch.Search;
using System;
using System.Collections.Generic;
using SabberStoneUtil.DataProcessing;

namespace DeckSearch
{
    class Program
    {
        static void Main(string[] args)
        {
            DataProcesser.PreprocessDeckData();
            // var search = new DistributedSearch(args[0]);
            // search.Run();
        }
    }
}
