using System;
using SurrogateModel.Surrogate;
using SabberStoneUtil.DataProcessing;


namespace SurrogateModel
{
    class Program
    {
        static void Main(string[] args)
        {
            var model = new FullyConnectedNN();
            model.OfflineFit();
        }
    }
}
