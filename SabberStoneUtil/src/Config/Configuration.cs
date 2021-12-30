namespace SabberStoneUtil.Config
{
    public class Configuration
    {
        public DeckspaceParams Deckspace { get; set; }
        public SearchParams Search { get; set; }
        public SurrogateParams Surrogate { get; set; }
    }

    public class DeckspaceParams
    {
        public string HeroClass { get; set; }
        public string[] CardSets { get; set; }
        public string[] AdditionalCards { get; set; }
    }

    public class SearchParams
    {
        public string Category { get; set; }
        public string Type { get; set; }
        public string ConfigFilename { get; set; }
        public int NumGeneration { get; set; }
        public int NumToEvaluatePerGeneration { get; set; }
        public int LogLengthPerGen { get; set; }
    }

    public class SurrogateParams
    {
        public string Type { get; set; }
        public string[] ModelTargets { get; set; }
        public string FixedModelSavePath { get; set; }
        public bool TestOutOfDist { get; set; }
        public string TestOutOfDistDataFile { get; set; }
    }
}
