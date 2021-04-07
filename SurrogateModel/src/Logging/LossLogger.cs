using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

// using MapSabber.Messaging;
using SabberStoneUtil.Decks;
using SabberStoneUtil.Messaging;

namespace SurrogateModel.Logging
{
    public class LossLogger
    {
        private string _logPath;
        private bool _isInitiated;

        public LossLogger(string logPath)
        {
            _logPath = logPath;
            _isInitiated = false;
        }

        private static void writeText(Stream fs, string s)
        {
            s += "\n";
            byte[] info = new UTF8Encoding(true).GetBytes(s);
            fs.Write(info, 0, info.Length);
        }

        /// <summary>
        /// Write header info to file
        /// </summary>
        private void initLog()
        {
            _isInitiated = true;

            // Create a log for individuals
            using (FileStream ow = File.Open(_logPath,
                      FileMode.Create, FileAccess.Write, FileShare.None))
            {
                string[] dataLabels = {
                    "train_loss",
                    "test_loss",
                };
                writeText(ow, string.Join(",", dataLabels));
                ow.Close();
            }
        }


        /// <summary>
        // Function to write a single individual
        /// </summary>
        public void LogLoss(double train_loss, double test_loss)
        {
            // Put the header on the log file if this is the first
            // individual in the experiment.
            if (!_isInitiated)
                initLog();

            using (StreamWriter sw = File.AppendText(_logPath))
            {
                string[] losses = {
                    train_loss.ToString(),
                    test_loss.ToString(),
                };
                sw.WriteLine(string.Join(",", losses));
                sw.Close();
            }
        }
    }
}
