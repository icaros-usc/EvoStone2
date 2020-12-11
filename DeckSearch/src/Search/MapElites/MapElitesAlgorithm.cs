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
      private int _individualsEvaluated;

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
      /// Another feature map used by Surrogated Search to record elites that are run for real
      /// </summary>
      FeatureMap _outerFeatureMap;

      // feature map loggers
      private FrequentMapLog _map_log;
      private FrequentMapLog _outer_map_log;

      public MapElitesAlgorithm(MapElitesParams config, string log_dir_exp)
      {
         _individualsDispatched = 0;
         _individualsEvaluated = 0;
         _params = config;

         // write the config file to the log directory for future reference
         string config_out_path = System.IO.Path.Combine(log_dir_exp, "elite_map_config.tml");
         Toml.WriteFile<MapElitesParams>(config, config_out_path);

         InitMap(log_dir_exp);
      }

      /// <summary>
      /// Initializes feature maps
      /// </summary>
      private void InitMap(string log_dir_exp)
      {
         var mapSizer = new LinearMapSizer(_params.Map.StartSize,
                                             _params.Map.EndSize);
         if (_params.Map.Type.Equals("SlidingFeature"))
         {
            _featureMap = new SlidingFeatureMap(_params.Search.NumToEvaluate, _params.Map, mapSizer);
            _outerFeatureMap = new SlidingFeatureMap(_params.Search.NumToEvaluate, _params.Map, mapSizer);
         }
         else if (_params.Map.Type.Equals("FixedFeature"))
         {
            _featureMap = new FixedFeatureMap(_params.Search.NumToEvaluate, _params.Map, mapSizer);
            _outerFeatureMap = new FixedFeatureMap(_params.Search.NumToEvaluate, _params.Map, mapSizer);
         }
         else
               Console.WriteLine("ERROR: No feature map specified in config file.");

         featureNames = new string[_params.Map.Features.Length];
         for (int i = 0; i < _params.Map.Features.Length; i++)
               featureNames[i] = _params.Map.Features[i].Name;


         // create logs
         string ELITE_MAP_FILENAME = System.IO.Path.Combine(log_dir_exp, "elite_map_log.csv");

         string OUTER_ELITE_MAP_FILENAME = System.IO.Path.Combine(log_dir_exp, "outer_elite_map_log.csv");

         _map_log = new FrequentMapLog(ELITE_MAP_FILENAME, _featureMap);
         _outer_map_log = new FrequentMapLog(OUTER_ELITE_MAP_FILENAME, _outerFeatureMap);
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

      public void ReturnEvaluatedIndividual(Individual cur)
      {
         cur.ID = _individualsEvaluated;
         _individualsEvaluated++;

         cur.Features = new double[featureNames.Length];
         for (int i = 0; i < featureNames.Length; i++)
            cur.Features[i] = cur.GetStatByName(featureNames[i]);

         _featureMap.Add(cur);
         _map_log.UpdateLog();
      }

      /// <summary>
      /// Add an individual to the _outerFeatureMap. Used by Surrogated Search
      /// </summary>
      public void AddToOuterFeatureMap(Individual cur)
      {
         _outerFeatureMap.Add(cur);
         _outer_map_log.UpdateLog();
      }

      /// <summary>
      /// Get all elites in the _featureMap
      /// </summary>
      public List<Individual> GetAllElites()
      {
         return _featureMap.EliteMap.Values.ToList();
      }
   }
}
