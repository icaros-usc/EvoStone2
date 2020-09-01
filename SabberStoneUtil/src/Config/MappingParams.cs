namespace SabberStoneUtil.Config
{
     public class MapParams
     {
          public string Type { get; set; }
          public int RemapFrequency { get; set; }
          public int StartSize { get; set; }
          public int EndSize { get; set; }

          public FeatureParams[] Features { get; set; }
     }

     public class FeatureParams
     {
          public string Name { get; set; }
          public double MinValue { get; set; }
          public double MaxValue { get; set; }
    }
}
