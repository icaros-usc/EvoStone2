using System;
using System.IO;
using System.Threading;

using SabberStoneUtil;
using SabberStoneUtil.Config;

namespace DeckSearch.Search
{
    /// <summary>
    /// Search without exploiting SurrogateModel
    /// </summary>
    class DistributedSearch
    {
        // Search Manager
        private DeckSearchManager _searchManager;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name = "configFilename">name of the configuation file</param>
        public DistributedSearch(string configFilename)
        {
            _searchManager = new DeckSearchManager(configFilename);
        }

        public void Run()
        {
            // let the workers know that searchAlgo is avialble
            _searchManager.AnnounceWorkersStart();

            Utilities.WriteLineWithTimestamp("Begin Distributed Search...");
            while (_searchManager.searchAlgo.IsRunning())
            {
                _searchManager.FindNewWorkers();
                _searchManager.DispatchSearchJobsToWorkers();
                _searchManager.FindDoneWorkers();
                Thread.Sleep(1000);
            }

            // Let the workers know that we are done.
            _searchManager.AnnounceWorkersDone();
        }
    }
}
