using CsvHelper.Configuration.Attributes;

namespace SabberStoneUtil.DataProcessing
{
    /// <summary>
    /// Includes deck features to be used for surrogate model
    /// </summary>
    public class ProcessedIndividual
    {
        [Name("Individual")]
        public int Index { get; set; }

        [Name("DeckManaSum")]
        public int DeckManaSum { get; set; }

        [Name("DeckManaVariance")]
        public double DeckManaVariance { get; set; }

        [Name("NumMinionCards")]
        public int NumMinionCards { get; set; }

        [Name("NumSpellCards")]
        public int NumSpellCards { get; set; }

        [Name("NumWeaponCards")]
        public int NumWeaponCards { get; set; }

        [Name("AttackSum")]
        public int AttackSum { get; set; }

        [Name("AttackVariance")]
        public double AttackVariance { get; set; }

        [Name("HealthSum")]
        public int HealthSum { get; set; }

        [Name("HealthVariance")]
        public double HealthVariance { get; set; }

        [Name("NumTaunt")]
        public int NumTaunt { get; set; }

        [Name("NumCharge")]
        public int NumCharge { get; set; }

        [Name("NumStealth")]
        public int NumStealth { get; set; }

        [Name("NumPoisonous")]
        public int NumPoisonous { get; set; }

        [Name("NumDivineShield")]
        public int NumDivineShield { get; set; }

        [Name("NumWindfury")]
        public int NumWindfury { get; set; }

        [Name("NumRush")]
        public int NumRush { get; set; }

        [Name("NumChooseOne")]
        public int NumChooseOne { get; set; }

        [Name("NumCombo")]
        public int NumCombo { get; set; }

        [Name("NumSecret")]
        public int NumSecret { get; set; }

        [Name("NumDeathrattle")]
        public int NumDeathrattle { get; set; }

        [Name("NumOverload")]
        public int NumOverload { get; set; }

        [Name("OverloadSum")]
        public int OverloadSum { get; set; }

        [Name("OverloadVariance")]
        public double OverloadVariance { get; set; }
    }
}