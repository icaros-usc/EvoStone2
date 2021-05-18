namespace SabberStoneUtil.Config
{
    public class RandomSearchParams
    {
        public RandomSearchSearchParams Search { get; set; }
        public MapParams Map { get; set; }
    }

    public class RandomSearchSearchParams
    {
        public int InitialPopulation { get; set; }
        public int NumToEvaluate { get; set; }
    }
}