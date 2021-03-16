using System;
using NumSharp;
using static Tensorflow.Binding;


namespace DeckSearch.Search.DataDrivenMapElites
{
    /// <summary>
    /// Implementation of UCB1 bandit algorithm.
    /// </summary>
    class UCB1
    {
        public NDArray probs;

        private int numOperator;

        private int numDist;

        /// <summary>
        /// Constructor of UCB1 algorithm.
        /// </summary>
        /// <param name="probs">2D NDArray of probability distributions of cross over operators.</param>
        public UCB1(NDArray probs)
        {
            this.probs = probs;
            numOperator = this.probs.shape[1];
            numDist = this.probs.shape[0];

        }

        public (int, NDArray) solve(NDArray successes, NDArray selections)
        {
            if ((successes.shape[0] != numDist) ||
                (selections.shape[0] != numDist))
            {
                throw new ArgumentException("Dimension of input does not match number of operators.");
            }
            NDArray decisons = successes / selections
                             + np.sqrt(2 * np.log(np.sum(successes.ToMuliDimArray<int>())) / selections);
            Console.Write("Decisons: ");
            print(decisons);
            int idx = np.argmax(decisons);
            return (idx, this.probs[idx]);
        }

    }
}