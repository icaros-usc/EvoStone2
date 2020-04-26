using System;
using System.Text;
using Tensorflow;
using NumSharp;
using static Tensorflow.Binding;

namespace DeckSearch.Surrogate
{
    /// <summary>
    /// DataLoader class. Used to store, preprocess, and sample training/testing data
    /// </summary>
    public class DataLoader
    {
        /// <summary>
        /// NDArray of features. Shape: [num_input, num_feature]
        /// </summary>
        private NDArray X = null;

        /// <summary>
        /// NDArray of target. Shape: [num_input, num_output]
        /// </summary>
        private NDArray y = null;

        /// <summary>
        /// Shuffle the data every epoch or not
        /// </summary>
        private bool shuffle = true;

        /// <summary>
        /// Counter to remember the index of sampling
        /// </summary>
        private int sample_idx = 0;

        /// <summary>
        /// Batch size of data
        /// </summary>
        private int batch_size = 0;

        /// <summary>
        /// Number of batches of the data
        /// </summary>
        public int num_batch {get; private set;} = 0;

        /// <summary>
        /// Constructor of DataLoader
        /// </summary>
        /// <param name="X">NDArray of features</param>
        /// <param name="y">NDArray of target</param>
        /// <param name="batch_size">batch size while sampling data. Default to 32</param>
        public DataLoader(NDArray X, NDArray y, int batch_size, bool shuffle = true)
        {
            if(X.shape[0] != y.shape[0])
            {
                throw new InvalidOperationException("DataLoader: shape[0] of X and y must match");
            }

            this.X = X;
            this.y = y;
            this.batch_size = batch_size;
            this.num_batch = X.shape[0] / batch_size;
            if(X.Shape[0] % batch_size != 0)
            {
                this.num_batch += 1;
            }

            // shuffle the data
            if(shuffle)
            {
                Shuffle();
            }
        }

        /// <summary>
        /// Function to sample one batch of data sequentially from X and y
        /// </summary>
        public (NDArray, NDArray)Sample()
        {
            // Console.WriteLine("sampling from [{0}, {1})\n", sample_idx*batch_size, (sample_idx+1)*batch_size);
            var x_sample = X[new Slice(sample_idx*batch_size, (sample_idx+1)*batch_size)];
            var y_sample = y[new Slice(sample_idx*batch_size, (sample_idx+1)*batch_size)];
            sample_idx += 1;

            // if all data are sampled, reset sample_idx and reshuffle for the next epoch
            if(sample_idx == num_batch)
            {
                sample_idx = 0;
                if(shuffle)
                {
                    Shuffle();
                }
            }

            return (x_sample, y_sample);
        }

        /// <summary>
        /// Function to shuffle rows of X and y
        /// Note: np.random.shuffle of NumSharp shuffle all cells instead of just the rows
        /// </summary>
        private void Shuffle()
        {
            var num_row = X.shape[0];
            int row_swap = -1;
            for (int i=num_row-1; i>0; i--)
            {
                // randomly choose row index to shuffle
                row_swap = np.random.randint(i);

                // swap the same rows of X and y
                Swap(X, row_swap, i);
                Swap(y, row_swap, i);
            }
        }

        /// <summary>
        /// Function to swap specified rows
        /// </summary>
        private void Swap(NDArray arr, int x, int y)
        {
            var temp = np.array(arr[x]).copy();
            arr[x] = arr[y];
            arr[y] = temp;
        }
    }
}