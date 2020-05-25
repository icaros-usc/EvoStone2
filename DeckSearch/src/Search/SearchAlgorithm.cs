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

      /// <summary>
      /// Record stats of the evaluated individual
      /// </summary>
      void ReturnEvaluatedIndividual(Individual ind);
   }
}
