using System;
using System.IO;
using System.Threading;

using DeckSearch.Config;

namespace DeckSearch.Search
{
    /// <summary>
    /// Search without exploiting SurrogateModel
    /// </summary>
    class DistributedSearch
    {
        // Search Manager
        private SearchManager _searchManager;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name = "configFilename">name of the configuation file</param>
        public DistributedSearch(Configuration config, string configFilename)
        {
            _searchManager = new SearchManager(config, configFilename);
        }

        public void Run()
        {
            // let the workers know that searchAlgo is avialble
            _searchManager.AnnounceWorkersStart();

            Console.WriteLine("Begin Distributed Search...");
            while (_searchManager._searchAlgo.IsRunning())
            {
                _searchManager.FindNewWorkers();
                _searchManager.DispatchJobsToWorkers();
                _searchManager.FindDoneWorkers();
                Thread.Sleep(1000);
            }

            // Let the workers know that we are done.
            _searchManager.AnnounceWorkersDone();
        }
    }
}
