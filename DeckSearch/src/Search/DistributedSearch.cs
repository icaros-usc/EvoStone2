using System;
using System.IO;
using System.Threading;

namespace DeckSearch.Search
{
    class DistributedSearch
    {
        // Search Manager
        private SearchManager _searchManager;

        public DistributedSearch(string configFilename)
        {
            _searchManager = new SearchManager(configFilename);
        }

        public void Run()
        {
            using (FileStream ow = File.Open(SearchManager._activeSearchPath,
                     FileMode.Create, FileAccess.Write, FileShare.None))
            {
                SearchManager.WriteText(ow, "MAP Elites");
                SearchManager.WriteText(ow, _searchManager._configFilename);
                ow.Close();
            }

            Console.WriteLine("Begin search...");
            while (_searchManager._searchAlgo.IsRunning())
            {
                _searchManager.FindNewWorkers();
                _searchManager.DispatchJobToWorkers();
                _searchManager.FindDoneWorkers();
                Thread.Sleep(1000);
            }

            // Let the workers know that we are done.
            File.Delete(SearchManager._activeSearchPath);
        }
    }
}
