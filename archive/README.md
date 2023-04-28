# CFD
A repository for computational fluid dynamics/heat transfer solvers 

The files in this repository are CFD programs written in python to solve heat transfer/fluid mechanics problems. 

1DHeatTransfer.py; a TDMA solver to solve1 dimensional heat transfer. 

FVsolver.py; a finite volume solver for heat transfer by diffusion and convection in a 2D control volume with dirichlet and neumann boundary conditions.

FVsolver-UWCD.py; A finite volume solver capable of computing upwind differencing method or centred differencing method to solve
heat transfer in a fluid of either rotating or linear velocity. The control volume has dirichlet boundary conditions.

SIMPLE.py; Semi-Implicit Method for Pressure Linked Equations solver to compute the classic lid driven cavity flow problem. 

A few pictures are included in this rep to illustrate model results. 
