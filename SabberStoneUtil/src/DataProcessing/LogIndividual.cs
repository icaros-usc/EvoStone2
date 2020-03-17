using CsvHelper.Configuration.Attributes;

namespace SabberStoneUtil.DataProcessing
{
    /// <summary>
    /// Used for reading in logged individual csv file
    /// </summary>
    public class LogIndividual
    {
        [Name("Individual")]
        public int IndividualID { get; set; }

        [Name("Parent")]
        public int Parent { get; set; }

        [Name("WinCount")]
        public int WinCount { get; set; }

        [Name("AverageHealthDifference")]
        public double AverageHealthDifference { get; set; }

        [Name("DamageDone")]
        public double DamageDone { get; set; }

        [Name("NumTurns")]
        public double NumTurns { get; set; }

        [Name("CardsDrawn")]
        public double CardsDrawn { get; set; }

        [Name("HandSize")]
        public double HandSize { get; set; }

        [Name("ManaSpent")]
        public double ManaSpent { get; set; }

        [Name("ManaWasted")]
        public double ManaWasted { get; set; }

        [Name("StrategyAlignment")]
        public double StrategyAlignment { get; set; }

        [Name("Dust")]
        public int Dust { get; set; }

        [Name("DeckManaSum")]
        public int DeckManaSum { get; set; }

        [Name("DeckManaVariance")]
        public double DeckManaVariance { get; set; }

        [Name("NumMinionCards")]
        public int NumMinionCards { get; set; }

        [Name("NumSpellCards")]
        public int NumSpellCards { get; set; }

        [Name("Deck")]
        public string Deck { get; set; }
    }
}