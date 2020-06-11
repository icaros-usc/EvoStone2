using System.Collections.Generic;

namespace SabberStoneUtil.DataProcessing
{
    public class CardEmbedding
    {
        public string cardName { get; set; }
        // public Dictionary<string, object> embedding;
        public double[] embedding { get; set; }
    }
}