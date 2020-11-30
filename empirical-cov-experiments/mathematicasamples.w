#!/Applications/Mathematica.app/Contents/MacOS/WolframKernel -script

(* This script generates samples with the WolframScript utility *)

parameters = If[Length[$ScriptCommandLine] == 0,
	Drop[$CommandLine,3],
	Drop[$ScriptCommandLine,1] 
];

If[Length[parameters] < 4, 
  Print["Usage: ./mathematicasamples.w <dim> <lower> <upper> <step> [<exponent code>] [<seed>]"]; Exit[]];

(*Print[$ScriptCommandLine];
Print[$CommandLine];
Print[parameters];
Print[parameters[[5]]];
Exit[];*)

dim = ToExpression[parameters[[1]]];
lower = ToExpression[parameters[[2]]];
upper = ToExpression[parameters[[3]]];
step = ToExpression[parameters[[4]]];

minExponent := 2.01;

maxExponent := 3.9;

(*
exponents := 
 Table[x, {x, minExponent, 
   maxExponent, (maxExponent - minExponent)/(dim - 1)}];
*)

exponents := Join[Table[6,{x,1,dim-2}],{2.1,2.1}];

If[Length[parameters] >= 5, 
	exponents = ToExpression[parameters[[5]]]
];

If[Length[parameters] >= 6,
    SeedRandom[ToExpression[parameters[[6]]]];
];

f[exp_][x_] := 1/(x^(exp));

norm[exp_] := NIntegrate[f[exp][Abs[x] + 1.5], {x, -Infinity, Infinity}];

P[exp_] := 
  ProbabilityDistribution[
   f[exp][Abs[x] + 1.5]/norm[exp], {x, -Infinity, Infinity}];

(* For an asymmetric distribution
f[exp_][x_] := 1/(x^(exp));

norm[exp_] := NIntegrate[f[exp][Abs[x-1] + 1.5], {x, -Infinity, Infinity}];

P[exp_] := 
  ProbabilityDistribution[
   f[exp][Abs[x-1] + 1.5]/norm[exp], {x, -Infinity, Infinity}];
*)

dist = ProductDistribution @@ Table[P[exp], {exp, exponents}];

thisfile = $InputFileName;
samplefolder = StringReplace[thisfile, "mathematicasamples.w" -> ""];

For[i = lower, i <= upper, i = i + step, X = RandomVariate[dist, i];
 Export[StringJoin[samplefolder, "samples/sample-", ToString[i], ".csv"], X, 
  "CSV"]];