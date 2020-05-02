using System;
using SurrogateModel.Surrogate;

namespace SurrogateModel
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            var model = new Model();
            model.OfflineFit();
        }
    }
}
