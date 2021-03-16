using DeckSearch.Search;
using SabberStoneUtil.Config;
using System.IO;
using System;

using Nett;

using SabberStoneUtil.DataProcessing;

using DeckSearch.Search.DataDrivenMapElites;
using NumSharp;
using static Tensorflow.Binding;

namespace DeckSearch
{
    class Program
    {
        static void Main(string[] args)
        {
            // // DataProcessor.GenerateCardDescription();

            // // read in config and initialize search space (domain of cards to search)
            // var config = Toml.ReadFile<Configuration>(args[0]);
            // CardReader.Init(config);

            // if(config.Search.Category == "Distributed")
            // {
            //     var search = new DistributedSearch(config, args[0]);
            //     search.Run();
            // }
            // else if(config.Search.Category == "Surrogated")
            // {
            //     var search = new SurrogatedSearch(config, args[0]);
            //     search.Run();
            // }

            NDArray probs = np.array(new double[,] {
                {0.5, 0.5}, {0.8, 0.2}, {1.0, 0}
            });

            var ucb1 = new UCB1(probs);
            // ucb1.solve();

            NDArray successes = np.zeros(new int[]{3});
            // successes[(int)np.random.randint(3)] = 1;
            NDArray selections = np.ones(new int[]{3});
            int num_iter = 100;

            Console.WriteLine("Initialization:");
            Console.Write("success:");
            print(successes);
            Console.Write("selections");
            print(selections);
            Console.WriteLine();

            // print(np.log(np.sum(successes.ToMuliDimArray<int>()))/ selections);

            for(int i = 0; i <num_iter; i++)
            {
                (int idx, NDArray curr_select) = ucb1.solve(successes, selections);
                Console.Write("curr selection: ");
                print(curr_select);


                // run "MAP-Elites" using the selected probability distribution
                // function for a generation of 100 solutions.
                selections[idx] += 100;
                successes[idx] += np.random.randint(101);

                Console.WriteLine("Iter {0}", i);
                Console.Write("success: ");
                print(successes);
                Console.Write("selections: ");
                print(selections);
                Console.WriteLine();
            }
        }
    }
}
