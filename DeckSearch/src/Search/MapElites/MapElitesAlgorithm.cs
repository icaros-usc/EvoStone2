using System;
using System.Linq;
using System.Collections.Generic;

using SabberStoneCore.Model;

using SabberStoneUtil.Config;

using Nett;

using DeckSearch.Logging;
using DeckSearch.Mapping;
using DeckSearch.Mapping.Sizers;

namespace DeckSearch.Search.MapElites
{
   class MapElitesAlgorithm : SearchAlgorithm
   {
      /// <summary>
      /// Parameters of the Map-Elites algorithm
      /// </summary>
      private MapElitesParams _params;

      /// <summary>
      /// Number of individuals estimated by evaluators
      /// </summary>
      public int _individualsEvaluated { get; private set; }

      /// <summary>
      /// Number of individuals dispatched to the evaluators
      /// </summary>
      private int _individualsDispatched;

      /// <summary>
      /// Name of the features in the feature map
      /// </summary>
      string[] featureNames;

      /// <summary>
      /// Feature map of the Map-Elites algorithm
      /// </summary>
      FeatureMap _featureMap;

      /// <summary>
      /// Another feature map used by Surrogated Search to record elites that
      /// are run on the surogate model
      /// </summary>
      FeatureMap _surrogateFeatureMap;

      // feature map loggers
      private FrequentMapLog _map_log;
      private FrequentMapLog _surrogate_map_log;


      // log directory
      private string _log_dir_exp;

      public MapElitesAlgorithm(MapElitesParams config, string log_dir_exp)
      {
         _individualsDispatched = 0;
         _individualsEvaluated = 0;
         _params = config;
         _log_dir_exp = log_dir_exp;

         // write the config file to the log directory for future reference
         string config_out_path = System.IO.Path.Combine(log_dir_exp, "elite_map_config.tml");
         Toml.WriteFile<MapElitesParams>(config, config_out_path);

         InitMaps(_log_dir_exp);
         InitLogs(_log_dir_exp);
      }

      /// <summary>
      /// Initializes feature maps
      /// </summary>
      private void InitMaps(string log_dir_exp)
      {
         var mapSizer = new LinearMapSizer(_params.Map.StartSize,
                                             _params.Map.EndSize);
         if (_params.Map.Type.Equals("SlidingFeature"))
         {
            _featureMap = new SlidingFeatureMap(_params.Search.NumToEvaluate, _params.Map, mapSizer);
            _surrogateFeatureMap = new SlidingFeatureMap(_params.Search.NumToEvaluate, _params.Map, mapSizer);
         }
         else if (_params.Map.Type.Equals("FixedFeature"))
         {
            _featureMap = new FixedFeatureMap(_params.Search.NumToEvaluate, _params.Map, mapSizer);
            _surrogateFeatureMap = new FixedFeatureMap(_params.Search.NumToEvaluate, _params.Map, mapSizer);
         }
         else
               Console.WriteLine("ERROR: No feature map specified in config file.");

         featureNames = new string[_params.Map.Features.Length];
         for (int i = 0; i < _params.Map.Features.Length; i++)
               featureNames[i] = _params.Map.Features[i].Name;

      }

      // create logs
      private void InitLogs(string log_dir_exp)
      {
         string ELITE_MAP_FILENAME = System.IO.Path.Combine(log_dir_exp, "elite_map_log.csv");

         string SURROGATE_ELITE_MAP_FILENAME = System.IO.Path.Combine(log_dir_exp, "surrogate_elite_map_log.csv");

         _map_log = new FrequentMapLog(ELITE_MAP_FILENAME, _featureMap);
         _surrogate_map_log = new FrequentMapLog(SURROGATE_ELITE_MAP_FILENAME, _surrogateFeatureMap);
      }

      public bool InitialPopulationEvaluated() => _individualsEvaluated >= _params.Search.InitialPopulation;

      public bool InitialPopulationDispatched() => _individualsDispatched >= _params.Search.InitialPopulation;

      public bool IsRunning() => _individualsEvaluated < _params.Search.NumToEvaluate;

      public bool IsBlocking() => _individualsDispatched >= _params.Search.InitialPopulation &&
                                 _individualsEvaluated == 0;

      public Individual GenerateIndividual(List<Card> cardSet)
      {
         _individualsDispatched++;
         return _individualsDispatched <= _params.Search.InitialPopulation ?
                Individual.GenerateRandomIndividual(cardSet) :
                _featureMap.GetRandomElite().Mutate();
      }

      public Individual GenerateIndividualFromSurrogateMap(List<Card> cardSet)
      {
         _individualsDispatched++;
         return _individualsDispatched <= _params.Search.InitialPopulation ?
                Individual.GenerateRandomIndividual(cardSet) :
                _surrogateFeatureMap.GetRandomElite().Mutate();
      }

      private void CalculateFeatures(Individual cur)
      {
         cur.Features = new double[featureNames.Length];
         for (int i = 0; i < featureNames.Length; i++)
            cur.Features[i] = cur.GetStatByName(featureNames[i]);
      }

      /// <summary>
      /// Add an individual to the _outerFeatureMap. Used by Surrogated Search
      /// </summary>
      public void AddToSurrogateFeatureMap(Individual cur)
      {
         CalculateFeatures(cur);
         _surrogateFeatureMap.Add(cur);
      }


      /// <summary>
      /// Add an individual to the _featureMap. Used by Surrogated Search
      /// </summary>
      public void AddToFeatureMap(Individual cur)
      {
         cur.ID = _individualsEvaluated;
         _individualsEvaluated++;

         CalculateFeatures(cur);

         _featureMap.Add(cur);
      }


      public void LogFeatureMap() {
         _map_log.UpdateLog();
      }

      public void LogSurrogateFeatureMap() {
         _surrogate_map_log.UpdateLog();
      }


      /// <summary>
      /// Get all elites in the _featureMap
      /// </summary>
      public List<Individual> GetAllElitesFromFeatureMap()
      {
         return _featureMap.EliteMap.Values.ToList();
      }

      /// <summary>
      /// Get all elites in the _featureMap
      /// </summary>
      public List<Individual> GetAllElitesFromSurrogateMap()
      {
         return _surrogateFeatureMap.EliteMap.Values.ToList();
      }


      /// <summary>
      /// Clear feature maps and create new ones.
      /// </summary>
      public void ClearMaps()
      {
         // init new maps
         InitMaps(_log_dir_exp);
         _individualsDispatched = 0;

         // update map logs
         _map_log.UpdateMap(_featureMap);
         _surrogate_map_log.UpdateMap(_surrogateFeatureMap);
      }


      public int NumIndividualsEvaled() { return this._individualsEvaluated; }
   }
}
