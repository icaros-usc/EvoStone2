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

namespace Analysis
{
    class Program
    {
        static void Main(string[] args)
        {
            var config = Toml.ReadFile<SabberStoneUtil.Config.Configuration>(args[0]);
            CardReader.Init(config);
        }
    }
}
