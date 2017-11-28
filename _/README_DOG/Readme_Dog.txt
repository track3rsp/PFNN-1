Input:
	dim:318
	input[0-191]   = B[129-320]  
								[0-71],   [129-200]        trajectory position, trajectory direction, 
								[72-95],  [201-224]        left and right heights
								[96-191], [225-320]		   gait 

	input[192-317] = A[3-128]   				           joints position, velocity 
	input[318]     = B[321]                                phase, but don't need to feed in post-process

Output:
	dim:154
	output[0-23] = C[165-200] 	(pick up 4 values every 6 value)    trajecoty positionX,Z velocityX,Z of future trajectory,
	output[24-149] = B[3-128]    									joints position, velocity
	output[150-153] = B[310:313]  RootTranslationalVelocityX,  RootTranslationalVelocityZ, RootAngularVelocity, PhaseChange


weight:

	W0_dim: 512*318
	B0_dim: 1*512
	W1_dim: 512*512
	B1_dim: 1*512
	W2_dim: 154*512
	B2_dim: 1*154
