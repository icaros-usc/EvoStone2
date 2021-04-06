using System;
using System.Linq;
using System.Threading;
using System.Collections.Generic;
using Nett;

using SurrogateModel.Surrogate;

using SabberStoneUtil.DataProcessing;
using SabberStoneUtil.Config;
using SabberStoneUtil.Decks;

using DeckEvaluator.Config;
using DeckEvaluator.Evaluation;

using SabberStoneCore.Enums;
using SabberStoneCore.Model;

using Tensorflow;
using NumSharp;
using static Tensorflow.Binding;

namespace PlayGround
{
    class Program
    {
        static void Main(string[] args)
        {
            // var config = Toml.ReadFile<Configuration>(args[0]);
            // CardReader.Init(config);
            // DataProcessor.GenerateCardDescription();
            // var model = new FullyConnectedNN();
            // var model = new DeepSetModel();
            // model.OfflineFit();
            // Console.WriteLine(DataProcessor.numCards);

            // // shape (3, 2, 2)
            // var a = new float[,,] { { { 1, 2 }, { 3, 4 } },
            //                         { { 5, 6 }, { 7, 8 } },
            //                         { { 9, 10 }, { 11, 12 } } };
            // var b = new float[,,] { { { 3, 4 }, { 1, 2 }},
            //                         { { 7, 8 }, { 5, 6 }, },
            //                         { { 11, 12 }, { 9, 10 },} };
            // var a_np = np.array(a);
            // var b_np = np.array(b);

            // Tensor x = tf.placeholder(tf.float32, shape: (3, 2, 2));

            // print(x.shape);
            // int[] real_shape = x.shape;
            // x = tf.reshape(x, new int[]{-1, 2});
            // print(real_shape);
            // print(x.shape);

            // // test fc3D
            // Tensor x = tf.placeholder(tf.float32, shape: (3, 2, 2));
            // var y = model.fc_layer(x, name: "fc3D", num_output: 4);
            // var sess = tf.Session();
            // sess.run(tf.global_variables_initializer());
            // var out_ = sess.run((y), (x, np.array(a)));
            // print(out_);
            // print(out_.shape);

            // // test deckEmbedding
            // var logIndividuals = DataProcessor.readLogIndividuals("resources/individual_log.csv");
            // var (deckEmbedding, deckStats) = DataProcessor.PreprocessDeckDataWithCard2VecEmbeddingFromData(logIndividuals.ToList());

            // var deckEmbedding_np = np.array(deckEmbedding);
            // print(deckEmbedding_np.shape);

            // // test tf.reduce_mean
            // Tensor z = tf.reduce_mean(x, axis: new int[]{1}, keepdims: true);
            // Tensor f = x - z;
            // var sess = tf.Session();
            // var (out_z, out_f) = sess.run((z, f), (x, np.array(a)));
            // print(np.array(a));
            // Console.WriteLine();
            // print(out_z);
            // print(out_z.shape);
            // Console.WriteLine();
            // print(out_f);
            // print(out_f.shape);

            // test per_equi_layer of DeepSet
            // DeepSetModel model = new DeepSetModel();
            // Tensor y = model.perm_equi_layer(x, "per_equi_max1", 4, "max");
            // var out_per_a = sess.run((y), (x, a_np));
            // print(out_per_a);
            // print(out_per_a.shape);

            // var out_per_b = sess.run((y), (x, b_np));
            // print(out_per_b);
            // print(out_per_b.shape);

            // test elu_layer on 3D input

            // DemoClassicCards(args);

            DemoTimeElapsed();
        }

        static public void DemoClassicCards(string[] args)
        {
            // ******* Some random demoing of classic cards *****
            // Card card = Cards.FromName("Execute");
            // // Card card = Cards.FromId("VAN_EX1_089");
            // Console.WriteLine("ID: {0}", card.Id);
            // Console.WriteLine("Cost: {0}", card.Cost);
            // Console.WriteLine("Attack: {0}", card.ATK);
            // Console.WriteLine("Health: {0}", card.Health);
            // Console.WriteLine("Text: {0}", card.Text);
            // Console.WriteLine("Implemented: {0}", card.Implemented);

            Console.WriteLine(args[0]);
            var config = Toml.ReadFile<DeckEvaluator.Config.Configuration>(args[0]);
            // Console.WriteLine(config.Nerfs.Length);
            int not_imp = 0;
            List<string> notImpCards = new List<string>();
            List<string> ImpCards = new List<string>();
            foreach (var nerfParam in config.Nerfs)
            {
                Card card = Cards.FromName(nerfParam.CardName);
                if (!card.Implemented) {
                    not_imp++;
                    Console.WriteLine("Name: {0}", card.Name);
                    Console.WriteLine("Cardset: {0}", card.Set);
                    Console.WriteLine("ID: {0}", card.Id);
                    Console.WriteLine("Cost: {0}", card.Cost);
                    Console.WriteLine("Attack: {0}", card.ATK);
                    Console.WriteLine("Health: {0}", card.Health);
                    Console.WriteLine("Text: {0}", card.Text);
                    Console.WriteLine("Implemented: {0}", card.Implemented);
                    Console.WriteLine();
                    notImpCards.add(card.Name);
                }
                else{
                    ImpCards.add(card.Name);
                    // if(ImpCards.Count >= 30) {
                    //     break;
                    // }
                }
            }
            Console.WriteLine("Not implemented: {0}", not_imp);

            // run a game with not implemented cards
            Deck playerDeck = new Deck("paladin", notImpCards.ToArray());
            Deck opponentDeck = new Deck("paladin", notImpCards.ToArray());

            var player = new PlayerSetup(playerDeck, PlayerSetup.GetStrategy("Control", null, null));
            var opponent = new PlayerSetup(opponentDeck, PlayerSetup.GetStrategy("Control", null, null));

            var gameEval = new GameEvaluator(player, opponent);
            var result = gameEval.PlayGame(1);
            Console.WriteLine(result._healthDifference);
        }

        static void DemoTimeElapsed()
        {
            DateTime start = DateTime.UtcNow;
            Thread.Sleep(2000);
            DateTime end = DateTime.UtcNow;
            TimeSpan timeDiff = end - start;
            Console.WriteLine(Convert.ToInt32(timeDiff.TotalMilliseconds));
        }
    }
}
