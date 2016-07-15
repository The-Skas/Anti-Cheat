using System;
using System.IO;
using DemoInfo;
using System.Diagnostics;
using System.Collections.Generic;

namespace DevNullPlayer
{
	class MainClass
	{
		public static void Main(string[] args)
		{
			Console.WriteLine ("Cool Stories.");
			using (var input = File.OpenRead(args[0])) {
				var parser = new DemoParser(input);
				
				parser.ParseHeader ();

			
				Dictionary<Player, int> failures = new Dictionary<Player, int>();
				parser.TickDone += (sender, e) => {
					//Problem: The HP coming from CCSPlayerEvent are sent 1-4 ticks later
					//I guess this is because the think()-method of the CCSPlayerResource isn't called
					//that often. Haven't checked though.
					foreach(var p in parser.PlayingParticipants)
					{


						if(p.HP < 100 && p.HP > 0) {
 							Console.WriteLine (p.Name + " HP:"+ p.HP);
						}
						//Okay, if it's wrong 2 seconds in a row, something's off
						//Since there should be a tick where it's right, right?
						//And if there's something off (e.g. two players are swapped)
						//there will be 2 seconds of ticks where it's wrong
						//So no problem here :)

					}
				};

				Dictionary<Player, int> killsThisRound = new Dictionary<Player, int> ();
				parser.PlayerKilled += (object sender, PlayerKilledEventArgs e) => {
					//the killer is null if you're killed by the world - eg. by falling
					if(e.Killer != null) {
						if(!killsThisRound.ContainsKey(e.Killer))
							killsThisRound[e.Killer] = 0;

						//Remember how many kills each player made this rounds
						killsThisRound[e.Killer]++;
					}
				};

				

				if (args.Length >= 2) {
					// progress reporting requested
					using (var progressFile = File.OpenWrite(args[1]))
					using (var progressWriter = new StreamWriter(progressFile) { AutoFlush = false }) {
						int lastPercentage = -1;
						while (parser.ParseNextTick()) {
							var newProgress = (int)(parser.ParsingProgess * 100);
							if (newProgress != lastPercentage) {
								progressWriter.Write(lastPercentage = newProgress);
								progressWriter.Flush();
							}
						}
					}

					return;
				}
		
				
					parser.ParseToEnd();
			}
		}
	}
}
