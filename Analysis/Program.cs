using System;

namespace Analysis
{
    class Program
    {
        static void Main(string[] args)
        {
            RemoveCardAnalysis rca = new RemoveCardAnalysis(args[0], args[1]);
            rca.Run();
        }
    }
}
