using System;
using System.Collections.Generic;
using System.Linq;
using System.Xml.Linq;

using SabberStoneCore.Enums;
using SabberStoneCore.Model;

namespace SabberStoneUtil.Config
{
    public class CardReader
    {
        /// <summary>
        /// Hero class read from SabberStone based on config file
        /// </summary>
        public static CardClass _heroClass { get; private set; }

        /// <summary>
        /// List of Cards read from SabberStone based on config file
        /// </summary>
        public static List<Card> _cardSet { get; private set; }

        /// <summary>
        /// static constructor to be called once to initialize the search space
        /// </summary>
        public static void Init(Configuration config)
        {
            // Configuration for the search space
            _heroClass = CardReader.GetClassFromName(config.Deckspace.HeroClass);
            CardSet[] sets = CardReader.GetSetsFromNames(config.Deckspace.CardSets);
            _cardSet = CardReader.GetCards(_heroClass, sets);

            // add specified additional card to the card set
            if (config.Deckspace.AdditionalCards != null)
            {
                foreach (string cardName in config.Deckspace.AdditionalCards)
                {
                    var matchCnt =
                        _cardSet.Where(c => c.Name == cardName).ToList().Count;
                    // add card if it does not exist in the current deck space
                    if (matchCnt == 0)
                    {
                        Card card = Cards.FromName(cardName);
                        _cardSet.Add(card);
                        Utilities.WriteLineWithTimestamp(
                            String.Format(
                                "Adding {0} from set {1} to search space.",
                                cardName, card.Set)
                        );
                    }
                }
            }
            Utilities.WriteLineWithTimestamp(
                String.Format("Number of cards in search space: {0}",
                              _cardSet.Count));
        }


        public static CardClass GetClassFromName(string className)
        {
            className = className.ToUpper();
            foreach (CardClass curClass in Cards.HeroClasses)
                if (curClass.ToString().Equals(className))
                    return curClass;

            Utilities.WriteLineWithTimestamp(
                "Card class " + className + " not a valid hero class.");
            return CardClass.NEUTRAL;
        }

        public static CardSet[] GetSetsFromNames(string[] cardSets)
        {
            Utilities.WriteLineWithTimestamp(
                "Using card sets: " + String.Join(" ", cardSets));
            var sets = new CardSet[cardSets.Length];

            for (int i = 0; i < cardSets.Length; i++)
                foreach (CardSet c in Enum.GetValues(typeof(CardSet)))
                {
                    if (c.ToString().Equals(cardSets[i]))
                        sets[i] = c;
                }
            return sets;
        }


        private static void printCards(List<Card> cards)
        {
            Console.WriteLine("Card set size = " + cards.Count);
            foreach (Card c in cards)
            {
                // Console.WriteLine(c.Name);
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
        }

        public static List<Card> GetCards(CardClass heroClass, CardSet[] sets)
        {
            var hs = new HashSet<Card>();
            var cards = new List<Card>();
            List<Card> allCards = Cards.All.ToList();
            foreach (Card c in allCards)
            {
                if (c != Cards.FromName("Default")
                    && c.Implemented
                    && c.Collectible
                    && c.Type != CardType.HERO
                    && c.Type != CardType.ENCHANTMENT
                    && c.Type != CardType.INVALID
                    && c.Type != CardType.HERO_POWER
                    && c.Type != CardType.TOKEN
                    && (c.Class == CardClass.NEUTRAL || c.Class == heroClass)
                    && !hs.Contains(c))
                {
                    bool validSet = false;
                    foreach (CardSet curSet in sets)
                        validSet |= (c.Set == curSet);

                    if (validSet)
                    {
                        cards.Add(c);
                        hs.Add(c);
                    }
                }
            }
            return cards;
        }
    }
}
