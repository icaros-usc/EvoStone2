using CsvHelper.Configuration.Attributes;


namespace SabberStoneUtil.DataProcessing
{
    /// <summary>
    /// A class used to write relevant card attributes to a csv file that will be used by card2vec model.
    /// </summary>
    public class LogCard
    {
        [Name("card_name")]
        public string cardName { get; set; }

        [Name("card_race")]
        public string cardRace { get; set; }

        [Name("card_type")]
        public string cardType { get; set; }

        [Name("card_class")]
        public string cardClass { get; set; }

        [Name("description")]
        public string description { get; set; }
    }
}