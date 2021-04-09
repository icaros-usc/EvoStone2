using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Text;

using DeckSearch.Mapping;
using DeckSearch.Search;

namespace DeckSearch.Logging
{
    // A compressed feature map for frequence logging.
    // Doesn't contain detailed individual information.
    class FrequentMapLog
    {
        private string _logPath;
        private FeatureMap _map;

        public FrequentMapLog(string logPath, FeatureMap map)
        {
            _logPath = logPath;
            _map = map;
            InitLog();
        }

        private void InitLog()
        {
            // Create a log for individuals
            using (FileStream ow = File.Open(_logPath,
                        FileMode.Create, FileAccess.Write, FileShare.None))
            {
                string[] dataLabels = {
                  "Dimensions",
                  "Map (f1xf2:CellCount:IndividualID:Wins:Fitness:Feature1:Feature2)"
               };

                WriteText(ow, string.Join(",", dataLabels));
                ow.Close();
            }
        }

        private static void WriteText(Stream fs, string s)
        {
            s += "\n";
            byte[] info = new UTF8Encoding(true).GetBytes(s);
            fs.Write(info, 0, info.Length);
        }

        // Call this whenever you want the log to update with the latest
        // feature map data.
        public void UpdateLog()
        {
            using (StreamWriter sw = File.AppendText(_logPath))
            {
                var rowData = new List<string>();
                IEnumerable<int> dimensions =
                   Enumerable.Repeat(_map.NumGroups, _map.NumFeatures);
                rowData.Add(string.Join("x", dimensions));

                foreach (string index in _map.EliteMap.Keys)
                {
                    Individual cur = _map.EliteMap[index];
                    var cellComponents = new List<string>();
                    cellComponents.Add(index);
                    cellComponents.Add(_map.CellCount[index].ToString());
                    cellComponents.Add(cur.ID.ToString());
                    cellComponents.Add(cur.OverallData.WinCount.ToString());
                    cellComponents.Add(cur.Fitness.ToString());
                    foreach (var curFeature in cur.Features)
                        cellComponents.Add(curFeature.ToString());
                    rowData.Add(string.Join(":", cellComponents));
                }

                sw.WriteLine(string.Join(",", rowData));
            }
        }

        // change the map to log
        public void UpdateMap(FeatureMap newMap, bool createNewLog=true)
        {
            // change map to log
            this._map = newMap;

			// if (createNewLog) {
			// 	// delete old log and initialize new one
			// 	File.Delete(_logPath);
			// 	InitLog();
			// }
        }
    }
}
