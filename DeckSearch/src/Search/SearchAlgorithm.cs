using System.Collections.Generic;
using SabberStoneCore.Model;

namespace DeckSearch.Search
{
   interface SearchAlgorithm
   {
      /// <summary>
      /// The algorithm is running until the number of evaluated individuals reaches a threshold
      /// </summary>
      bool IsRunning();

      /// <summary>
      /// The algorithm is blocking when none of the individuals in the initial population is finished with evaluation
      /// </summary>
      bool IsBlocking();

      /// <summary>
      /// The algorithm reaches the initial population when the number of evaluated individuals is greater or equal to the initial population size
      /// </summary>
      bool InitialPopulationEvaluated();

      /// <summary>
      /// The algorithm has dispatched enough individuals to construct initial population.
      bool InitialPopulationDispatched();

      /// <summary>
      /// Generate a random individual or mutate a current individual
      /// </summary>
      Individual GenerateIndividual(List<Card> cardSet);


      // ***********************************
      // Optional methods below
      // ***********************************

      /// <summary>
      /// Record stats of the evaluated individual and add an individual to the
      /// _featureMap. Used by Surrogated Search.
      /// </summary>
      void AddToFeatureMap(Individual ind) {}

      /// <summary>
      /// Generate a random individual or mutate a current individual
      /// </summary>
      Individual GenerateIndividualFromSurrogateMap(List<Card> cardSet)
      { return null; }

      /// <summary>
      /// Add an individual to the _outerFeatureMap. Used by Surrogated Search
      /// </summary>
      void AddToSurrogateFeatureMap(Individual cur) {}

      /// <summary>
      /// Log content of feature maps to disk.
      /// </summary>
      void LogFeatureMap() {}

      /// <summary>
      /// Log content of surrogate feature maps to disk.
      /// </summary>
      void LogSurrogateFeatureMap() {}

      /// <summary>
      /// Get all elites in the _featureMap
      /// </summary>
      List<Individual> GetAllElitesFromFeatureMap() {return null;}

      /// <summary>
      /// Get all elites in the _featureMap
      /// </summary>
      List<Individual> GetAllElitesFromSurrogateMap() {return null;}


      int NumIndividualsEvaled() { return 0; }

   }
}
