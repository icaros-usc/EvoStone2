using System;
using SurrogateModel.Surrogate;
using SabberStoneUtil.DataProcessing;


namespace SurrogateModel
{
    class Program
    {
        private const string TRAIN_LOG_DIRECTORY = "train_log/";

        private const string TRAINING_DATA_LOG_FILENAME_PREFIX =
           TRAIN_LOG_DIRECTORY + "training_data";

        // record the index of the training data file
        private static int training_idx = 0;

        static void Main(string[] args)
        {
            var model = new Model();
            Console.WriteLine("Computation Graph is built");

            while(true)
            {
                // obtain the data
                (int[,] cardsEncoding, double[,] deckStats) = DataProcessor.ReadNewTrainingData(TRAINING_DATA_LOG_FILENAME_PREFIX + training_idx.ToString() + ".csv");
                Console.WriteLine("Data obtained, start back prop");
                model.OnlineFit(cardsEncoding, deckStats);
                training_idx++; // increment index for next the next loop
            }
        }
    }
}
