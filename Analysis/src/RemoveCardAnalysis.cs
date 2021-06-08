using System;
using System.Threading;

namespace Analysis
{
    public class RemoveCardAnalysis
    {
        private RemoveCardAnalysisManager _rcaManager;
        public RemoveCardAnalysis(
            string expLogDir,
            string configFilename)
        {
            _rcaManager =
                new RemoveCardAnalysisManager(expLogDir, configFilename);
        }

        public void Run()
        {
            _rcaManager.AnnounceWorkersStart();

            Console.WriteLine("Begin Remove Card Analysis");
            while(_rcaManager.Running())
            {
                _rcaManager.FindNewWorkers();
                _rcaManager.DispatchEvalJobsToWorkers();
                _rcaManager.FindDoneWorkers();
                _rcaManager.FindOvertimeWorkers();
                Thread.Sleep(1000);
            }

            _rcaManager.AnnounceWorkersDone();
        }
    }
}