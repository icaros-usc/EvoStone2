using DeckSearch.Search;
using DeckSearch.Config;

using Nett;

using SabberStoneUtil.DataProcessing;

namespace DeckSearch
{
    class Program
    {
        static void Main(string[] args)
        {
            DataProcessor.GenerateCardDescription();
            // var config = Toml.ReadFile<Configuration>(args[0]);
            // if(config.Search.Category == "Distributed")
            // {
            //     var search = new DistributedSearch(config, args[0]);
            //     search.Run();
            // }
            // else if(config.Search.Category == "Surrogated")
            // {
            //     var search = new SurrogatedSearch(config, args[0]);
            //     search.Run();
            // }
        }
    }
}
