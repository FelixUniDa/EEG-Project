Comparison of PowerICA with pow3 nonlinearity and GraphPowerICA with differing Parameters:

- Signaltype:		Graph Gaussian Moving Average of 3 Standard-t distributions and one normal distribution
			Standardtype (ECG, cosine, sawtooth and square) -> Graph from Correlationmatrix of time instances
- Samplesize:		1000 samples
- Nonlinearities:	[arctan, gaus, tanh, pow3]
- b(1-lambda):		[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
- Monte-Carlo-Runs: 	100


Missing Boxplots are due to the algorithm having convergence problems, which rules out those parameters anyway.