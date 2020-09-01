namespace SabberStoneUtil.Config
{
   public class EvolutionStrategyParams
   {
      public EvolutionStrategySearchParams Search { get; set; }
   }

   public class EvolutionStrategySearchParams
   {
      public int InitialPopulation { get; set; }
      public int NumToEvaluate { get; set; }
      public int NumParents { get; set; }
   }
}
