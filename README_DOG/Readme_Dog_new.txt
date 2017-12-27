Input:
	dim_python:505, dim_unity:504
	input[0-179]   = B[327-506]  
								[0-71],   [327-398]        trajectory position, trajectory direction
								[72-95],  [399-422]        left and right heights
								[96-179], [423-506]		   gait 

	input[180-503] = A[3-326]   				           joints position, rotation, velocity
	input[504]     = B[507]                                phase, but don't need to feed in post-process

Output:
	dim_python:352, dim_unity:352
	output[0-23] = C[363-398] 	 (pick up 4 values every 6 value)    trajectory positionX,Z velocityX,Z of future trajectory,
	output[24-347] = B[3-326]    									joints position, rotation, velocity,
	output[348-351] = B[508:511]  RootTranslationalVelocityX,  RootTranslationalVelocityZ, RootAngularVelocity, PhaseChange


weight:

	W0_dim: 512*505
	B0_dim: 1*512
	W1_dim: 512*512
	B1_dim: 1*512
	W2_dim: 352*512
	B2_dim: 1*352
