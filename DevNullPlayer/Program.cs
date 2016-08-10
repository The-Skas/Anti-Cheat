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
			
				//Get map name
				string map = parser.Map;
				// And now, generate the filename of the resulting file
				string[] _output = args[0].Split(new[]{"/","."},StringSplitOptions.None);
				string demo_name = _output [_output.Length - 2];
				string outputFileName = Math.Round(parser.TickRate)+"t_" + map+"_" +demo_name+ ".csv";
				// and open it. 
				var outputStream = new StreamWriter (outputFileName);
				//Write to csv file headers first:
				//Write Header? Possible Issue is if Im writing multiple files so organising which is which is good.
				outputStream.WriteLine (WriteCSVLine("Steam_ID", "Name","Tick" ,"Time", "Round", "Alive", "X","Y", "Z", "VelX","VelY","VelZ", "ViewX", "ViewY", 
													"ViewXPunchAngle", "ViewYPunchAngle", "AimXPunchAngle", "AimYPunchAngle", "AimXPunchVel", "AimYPunchVel",
													"ViewZOffset", "HasShot", "Weapon"));

				string outputFileHurt = Math.Round(parser.TickRate)+"t_" + map+"_" +demo_name+ "_"+"attackinfo.csv";
				var outputHurtStream = new StreamWriter (outputFileHurt);
				outputHurtStream.WriteLine(WriteCSVLine ("Tick","Attacker", "Victim" ,"HitGroup", "Weapon"));
				//PARSING GOES HERE

				int round_total = 0;
				int roundCSV = 0;
				Dictionary<Player, int> failures = new Dictionary<Player, int>();
				parser.TickDone += (sender, e) => {
					//Problem: The HP coming from CCSPlayerEvent are sent 1-4 ticks later
					//I guess this is because the think()-method of the CCSPlayerResource isn't called
					//that often. Haven't checked though.
					foreach(var p in parser.PlayingParticipants)
					{
						
						var wep = "None";
						if(p.ActiveWeapon != null && p.ActiveWeapon.OriginalString != null) {
							wep = p.ActiveWeapon.Weapon.ToString();
						}
						// ID ; Tick ; Time ;
						outputStream.WriteLine (WriteCSVLine(p.SteamID,p.Name, parser.IngameTick, parser.CurrentTime, roundCSV, p.IsAlive, p.Position.X, p.Position.Y, p.Position.Z, p.Velocity.X, p.Velocity.Y, p.Velocity.Z,  p.ViewDirectionX, p.ViewDirectionY,
							p.ViewPunchAngle.X, p.ViewPunchAngle.Y, p.AimPunchAngle.X, p.AimPunchAngle.Y, p.AimPunchVel.X, p.AimPunchVel.Y, 
							p.ViewOffsetZ, p.HasShot, wep )); 


						//Okay, if it's wrong 2 seconds in a row, something's off
						//Since there should be a tick where it's right, right?
						//And if there's something off (e.g. two players are swapped)
						//there will be 2 seconds of ticks where it's wrong
						//So no problem here :)
						p.HasShot = false;

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

				parser.PlayerHurt += (object sender, PlayerHurtEventArgs e) => {
					//TODO: Test if The attacker is null if the world damages the player?
					if(e.Attacker != null) {
						int health = e.Health;
						outputHurtStream.WriteLine(WriteCSVLine(parser.IngameTick, e.Attacker.SteamID, e.Player.SteamID ,e.Hitgroup.ToString(), e.Weapon.Weapon.ToString()));

					} else {
						outputHurtStream.WriteLine(WriteCSVLine(parser.IngameTick, "World", e.Player.SteamID ,e.Hitgroup.ToString(), "None"));
					}

						
				};


				parser.FreezetimeEnded    += (object sender, FreezetimeEndedEventArgs e) => {
					round_total += 1;
					roundCSV = round_total;
				};
				parser.RoundOfficiallyEnd += (object sender, RoundOfficiallyEndedEventArgs e) => {
					roundCSV = 0;
				};

				parser.WeaponFired += (object sender, WeaponFiredEventArgs e) => {
					// e.Shooter.ActiveWeapon;
					e.Shooter.HasShot = true;
					//var _tick = parser.IngameTick;
					//var bleh_0 = 0;
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
				//END: Closes File
				outputStream.Close ();
				outputHurtStream.Close();
				int x = 5;
				x += 2;
			}
		}



		static string WriteCSVLine(params object[] args)
		{
			string result = "";
			foreach (object arg in args) {
				if (arg == null) {
					throw new ArgumentNullException ();
				}
				result += arg.ToString() +";";

			}
			return result.Substring(0,result.Length - 1);
//			return string.Format(
//				"{0};{1};{2};{3};{4};{5};{6};{7};{8};{9};{10};{11};{12};{13};{14};{15};{16};{17};{18};{19};{20};{21};{22};{23};",
//				"Round-Number", // parser.CTScore + parser.TScore, //Round-Number
//				"CT-Score", // parser.CTScore,
//				"T-Score", // parser.TScore,
//				//how many CTs are still alive?
//				"SurvivingCTs", // parser.PlayingParticipants.Count(a => a.IsAlive && a.Team == Team.CounterTerrorist),
//				//how many Ts are still alive?
//				"SurvivingTs", // parser.PlayingParticipants.Count(a => a.IsAlive && a.Team == Team.Terrorist),
//				"CT-StartMoney", // ctStartroundMoney,
//				"T-StartMoney", // tStartroundMoney,
//				"CT-EquipValue", // ctEquipValue,
//				"T-EquipValue", // tEquipValue,
//				"CT-SavedFromLastRound", // ctSaveAmount,
//				"T-SavedFromLastRound", // tSaveAmount,
//				"WalkedCTWay", // ctWay,
//				"WalkedTWay", // tWay,
//				//The kills of all CTs so far
//				"CT-Kills", // parser.PlayingParticipants.Where(a => a.Team == Team.CounterTerrorist).Sum(a => a.AdditionaInformations.Kills),
//				"T-Kills", // parser.PlayingParticipants.Where(a => a.Team == Team.Terrorist).Sum(a => a.AdditionaInformations.Kills),
//				//The deaths of all CTs so far
//				"CT-Deaths", // parser.PlayingParticipants.Where(a => a.Team == Team.CounterTerrorist).Sum(a => a.AdditionaInformations.Deaths),
//				"T-Deaths", // parser.PlayingParticipants.Where(a => a.Team == Team.Terrorist).Sum(a => a.AdditionaInformations.Deaths),
//				//The assists of all CTs so far
//				"CT-Assists", // parser.PlayingParticipants.Where(a => a.Team == Team.CounterTerrorist).Sum(a => a.AdditionaInformations.Assists),
//				"T-Assists", // parser.PlayingParticipants.Where(a => a.Team == Team.Terrorist).Sum(a => a.AdditionaInformations.Assists),
//				"BombPlanted", // plants,
//				"BombDefused", // defuses,
//				"TopfraggerName", // "\"" + topfragger.Key.Name + "\"", //The name of the topfragger this round
//				"TopfraggerSteamid", // topfragger.Key.SteamID, //The steamid of the topfragger this round
//				"TopfraggerKillsThisRound" // topfragger.Value //The amount of kills he got
//			);
		}
	}
}
