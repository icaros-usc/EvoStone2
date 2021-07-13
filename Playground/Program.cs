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
            var config = Toml.ReadFile<SabberStoneUtil.Config.Configuration>(args[0]);
            CardReader.Init(config);
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

            // DemoTimeElapsed();

            // DemoTFSaver();

            // DemoDeepSet();

            // Demo2DSlicing();

            // DemoGradient();

            // DemoSavedModel();

            // DataProcessor.WriteCardIndex();

            // DemoToml(args[0]);


            // var deck = new List<string>() { "Backstab", "Backstab", "Preparation", "Preparation", "Shadowstep", "Shadowstep", "Cold Blood", "Cold Blood", "Conceal", "Conceal", "Deadly Poison", "Deadly Poison", "Blade Flurry", "Eviscerate", "Eviscerate", "Sap", "Shiv", "Shiv", "Edwin VanCleef", "Fan of Knives", "Fan of Knives", "SI:7 Agent", "SI:7 Agent", "Bloodmage Thalnos", "Earthen Ring Farseer", "Leeroy Jenkins", "Azure Drake", "Azure Drake", "Gadgetzan Auctioneer", "Gadgetzan Auctioneer" };
            // PrintDeckInfo(deck);
        }

        static public void PrintCardInfo(Card c)
        {
            Console.WriteLine("Name: {0}", c.Name);
            Console.WriteLine("ID: {0}", c.Id);
            Console.WriteLine("Cost: {0}", c.Cost);
            Console.WriteLine("ATK: {0}", c.ATK);
            Console.WriteLine("Health: {0}", c.Health);
            Console.WriteLine("Card Set: {0}", c.Set);
            Console.WriteLine("Card Class: {0}", c.Class);
            Console.WriteLine("Text: {0}", c.Text);
            Console.WriteLine("Implemented: {0}", c.Implemented);
            Console.WriteLine("-----------------------------");
        }


        static public void PrintDeckInfo(List<string> deck)
        {
            foreach (var cardName in deck)
            {
                Card card = Cards.FromName(cardName);
                Console.WriteLine("Name: {0}", card.Name);
                Console.WriteLine("Cost: {0}", card.Cost);
                Console.WriteLine("ATK: {0}", card.ATK);
                Console.WriteLine("Health: {0}", card.Health);
                Console.WriteLine("Class: {0}", card.Set);
                Console.WriteLine("Text: {0}", card.Text);
                Console.WriteLine("Implemented: {0}", card.Implemented);
                Console.WriteLine("-----------------------------");
            }
            Console.WriteLine("Total cards: {0}", deck.Count);
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
                if (!card.Implemented)
                {
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
                else
                {
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

        static void DemoTFSaver()
        {
            // save
            var model = new FullyConnectedNN();
            model.OfflineFit();

            // // restore
            // var model = new FullyConnectedNN();
            // model.LoadModel("train_log/model.ckpt");
            // model.OfflineFit();
        }

        static void DemoSavedModel()
        {
            // save
            // var model = new FullyConnectedNN();
            // model.OfflineFit();

            // restore
            // var model = new LinearModel();
            // model.LoadModel("logs/2021-05-18_15-16-41_Surrogated_MAP-Elites_LinearModel_10000/surrogate_train_log/surrogate_model/model0/model.ckpt");


            var model = new DeepSetModel();
            model.LoadModel("logs/2021-04-22_01-14-27_Surrogated_MAP-Elites_DeepSetModel_10000/surrogate_train_log/surrogate_model/model18/model.ckpt");

            print(model.Forward(np.ones(new int[] { 2, 30, 178 })));
            // model.OfflineFit();
        }

        // test per_equi_layer of DeepSet
        static void DemoDeepSet()
        {
            var model = new DeepSetModel();
            var sess = tf.Session();

            // Tensor x = tf.placeholder(tf.float32, shape: (3, 2, 2));
            // Tensor y = model.perm_equi_layer(x, "per_equi1", 4, "mean");
            // sess.run(tf.global_variables_initializer());

            // // shape (3, 2, 2)
            // var a = new float[,,] { { { 1, 2 }, { 3, 4 } },
            //                         { { 5, 6 }, { 7, 8 } },
            //                         { { 9, 10 }, { 11, 12 } } };
            // var a_np = np.array(a);
            // var out_per_a = sess.run((y), (x, a_np));
            // print(out_per_a);
            // print(out_per_a.shape);
        }

        static void Demo2DSlicing()
        {
            var a = np.random.rand(new int[] { 2, 2 });
            print(a);
            print(a[Slice.All, Slice.Index(1)]);
        }

        static void DemoGradient()
        {
            var model = new DeepSetModel();
            // var sess = tf.Session();

            var x_input = np.random.rand(new int[] { 2, 30, 178 });
            // print(model.Forward(x_input));

            print(model.input);
            print(model.model_output);

            print(model.TakeGradient(x_input));

            // using var g = tf.GradientTape();

            // print(gradients);

            // var a = tf.constant(1f);
            // var b = tf.tanh(a);
            // var g = tf.gradients(b, a);
            // using (var sess = tf.Session())
            // {
            //     var result = sess.run(g);
            //     var actual = result[0].GetData<float>()[0];
            //     print(result);
            //     // self.assertEquals(0.41997434127f, actual);
            // }
        }

        static void DemoToml(string configFileName)
        {
            var evalConfig = Toml.ReadFile<DeckEvaluator.Config.Configuration>(configFileName);
            Console.WriteLine(evalConfig.Evaluation.PlayerStrategies[0].Weights == null);
        }
    }
}
