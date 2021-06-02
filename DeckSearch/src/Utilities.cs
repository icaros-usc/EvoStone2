using System.Runtime.Serialization.Formatters.Binary;
using System.IO;
using System;

namespace DeckSearch
{
    public static class Utilities
    {
        public static T DeepClone<T>(this T obj)
        {
            using (var ms = new MemoryStream())
            {
                var formatter = new BinaryFormatter();
                formatter.Serialize(ms, obj);
                ms.Position = 0;

                return (T)formatter.Deserialize(ms);
            }
        }

        public static void WriteLineWithTimestamp(string line)
        {
            string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            Console.WriteLine(String.Format("{0} | {1}", timestamp, line));
        }
    }
}