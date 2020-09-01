namespace SabberStoneUtil.Config
{
    public class MapElitesParams
    {
        public MapElitesSearchParams Search { get; set; }
        public MapParams Map { get; set; }
    }

    public class MapElitesSearchParams 
    {
        public int InitialPopulation { get; set; }
        public int NumToEvaluate { get; set; }
    }
}