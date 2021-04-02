namespace DeckEvaluator.Config
{
   public class Configuration
   {
      public EvaluationParams Evaluation { get; set; }
      public NetworkParams Network { get; set; }
      public NerfParams[] Nerfs { get; set; }
   }

   public class EvaluationParams
   {
      public string OpponentDeckSuite { get; set; }
      public string[] DeckPools { get; set; }
      public PlayerStrategyParams[] PlayerStrategies { get; set; }
   }

   public class PlayerStrategyParams
   {
      public int NumGames { get; set; }
      public string Strategy { get; set; }
   }

   public class NetworkParams
   {
      public int[] LayerSizes { get; set; }
   }

   public class NerfParams
   {
      public string CardName { get; set; }
      public int NewManaCost { get; set; }
      public int NewAttack { get; set; }
      public int NewHealth { get; set; }
   }
}
