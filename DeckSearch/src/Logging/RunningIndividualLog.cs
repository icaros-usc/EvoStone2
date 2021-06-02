using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

// using MapSabber.Messaging;
using SabberStoneUtil.Decks;
using SabberStoneUtil.Messaging;
using DeckSearch.Search;

namespace DeckSearch.Logging
{
   public class RunningIndividualLog
   {
      private string _logPath;
      private bool _isInitiated;

      public RunningIndividualLog(string logPath)
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
      private void initLog(Individual cur)
      {
         _isInitiated = true;

         // Create a log for individuals
         using (FileStream ow = File.Open(_logPath,
                   FileMode.Create, FileAccess.Write, FileShare.None))
         {
            var dataLabels = GetDataLabels(cur);
            writeText(ow, string.Join(",", dataLabels));
            ow.Close();
         }
      }

      /// <summary>
      /// Helper function to get the headers of the log file
      /// </summary>
      private IEnumerable<string> GetDataLabels(Individual cur)
      {
         // The data to maintain for individuals evaluated.
         string[] individualLabels = {
               "Individual",
               "Parent"
            };

         string[] deckLabels = {
               "Deck"
            };

         var dataLabels = individualLabels
            .Concat(OverallStatistics.Properties);
         for(int i=0; i<cur.StrategyData.Length; i++)
         {
            string prefix = String.Format("S{0}:", i);

            var strategyLabels =
               StrategyStatistics.Properties
               .Select(x => prefix+x);
            dataLabels = dataLabels.Concat(strategyLabels);
         }

         dataLabels = dataLabels.Concat(deckLabels);
         return dataLabels;
      }

      /// <summary>
      // Function to write a single individual
      /// </summary>
      public void LogIndividual(Individual cur)
      {
         // Put the header on the log file if this is the first
         // individual in the experiment.
         if (!_isInitiated)
            initLog(cur);

			using (StreamWriter sw = File.AppendText(_logPath))
         {
            var data = GetIndividualData(cur);
            sw.WriteLine(string.Join(",", data));
            sw.Close();
         }
      }

      /// <summary>
      // Function to write a list of individuals
      /// </summary>
      public void LogIndividuals(List<Individual> individuals)
      {
         // locks the file
         using (FileStream ow = File.Open(_logPath,
                   FileMode.Create, FileAccess.Write, FileShare.None))
         using (StreamWriter sw = new StreamWriter(ow))
         {
            // Put the header in the log file
            var dataLabels = GetDataLabels(individuals[0]);
            writeText(ow, string.Join(",", dataLabels));

            foreach (var cur in individuals)
            {
               var data = GetIndividualData(cur);
               sw.WriteLine(string.Join(",", data));
            }
            sw.Close();
            ow.Dispose(); // unlocks the file after writing
         }
      }

      /// <summary>
      /// Function to get the string data to write to files
      /// </summary>
      public IEnumerable<string> GetIndividualData(Individual cur)
      {
         string[] individualData = {
               cur.ID.ToString(),
               cur.ParentID.ToString(),
            };

         var overallStatistics =
            OverallStatistics.Properties
            .Select(x => cur.OverallData.GetStatByName(x).ToString());
         var data = individualData.Concat(overallStatistics);
         foreach (var stratData in cur.StrategyData)
         {
            var strategyData = StrategyStatistics.Properties
               .Select(x => stratData.GetStatByName(x).ToString());
            data = data.Concat(strategyData);
         }

         string[] deckData = {
               string.Join("*", cur.GetCards())
            };
         data = data.Concat(deckData);
         return data;
      }
   }
}
