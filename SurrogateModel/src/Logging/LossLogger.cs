using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using NumSharp;

// using MapSabber.Messaging;
using SabberStoneUtil.Decks;
using SabberStoneUtil.Messaging;

namespace SurrogateModel.Logging
{
    public class LossLogger
    {
        private string _logPath;
        private bool _isInitiated;
        private string[] _model_targets;

        public LossLogger(string logPath, string[] model_targets)
        {
            _logPath = logPath;
            _isInitiated = false;
            _model_targets = model_targets;
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
        private void initLog(bool testOutOfDist)
        {
            _isInitiated = true;

            // Create a log for individuals
            using (FileStream ow = File.Open(_logPath,
                      FileMode.Create, FileAccess.Write, FileShare.None))
            {
                List<string> dataLabels = new List<string>(){
                    "Sum train loss",
                    "Sum test loss",
                };

                if (testOutOfDist)
                {
                    dataLabels.Add("Sum test out-of-dist loss");
                }

                foreach (string target in _model_targets)
                {
                    dataLabels.Add(target + " train loss");
                    dataLabels.Add(target + " test loss");

                    if (testOutOfDist)
                    {
                        dataLabels.Add(target + " test out-of-dist loss");
                    }

                }
                writeText(ow, string.Join(",", dataLabels));
                ow.Close();
            }
        }


        /// <summary>
        // Function to write a single individual
        /// </summary>
        public void LogLoss(
            double train_loss,
            double test_loss,
            double test_loss_out_dist,
            NDArray train_per_ele_loss,
            NDArray test_per_ele_loss,
            NDArray test_per_ele_loss_out_dist)
        {
            bool testOutOfDist = false;
            if (!Double.IsNaN(test_loss_out_dist) &&
                !test_per_ele_loss_out_dist.Equals(null))
            {
                testOutOfDist = true;
            }
            // Put the header on the log file if this is the first
            // individual in the experiment.
            if (!_isInitiated)
                initLog(testOutOfDist);

            using (StreamWriter sw = File.AppendText(_logPath))
            {
                List<string> losses = new List<string>(){
                    train_loss.ToString(),
                    test_loss.ToString(),
                };

                // Add out-of-dist loss.
                if (testOutOfDist)
                {
                    losses.Add(test_loss_out_dist.ToString());
                }
                for (int i = 0; i < train_per_ele_loss.shape[0]; i++)
                {
                    var train_ele_loss = train_per_ele_loss[i];
                    var test_ele_loss = test_per_ele_loss[i];
                    losses.Add(train_ele_loss.ToString());
                    losses.Add(test_ele_loss.ToString());

                    // Add out-of-dist loss.
                    if (testOutOfDist)
                    {
                        var test_ele_loss_out_dist =
                            test_per_ele_loss_out_dist[i];
                        losses.Add(test_ele_loss_out_dist.ToString());
                    }
                }

                sw.WriteLine(string.Join(",", losses));
                sw.Close();
            }
        }
    }
}
