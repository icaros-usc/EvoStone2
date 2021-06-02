using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Collections.Concurrent;
using Nett;

using SurrogateModel.Surrogate;

using SabberStoneUtil.DataProcessing;
using SabberStoneUtil.Config;
using SabberStoneUtil.Decks;
using SabberStoneUtil.Messaging;

using DeckSearch.Search;

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
            RemoveCardAnalysis rca = new RemoveCardAnalysis(args[0], args[1]);
            rca.Run();
        }
    }
}
