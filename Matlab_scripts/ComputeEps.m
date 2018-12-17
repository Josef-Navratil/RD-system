function [eps0]=ComputeEps(HatC,rhoS,normB,C)
eps0=(-(rhoS+normB)+sqrt((rhoS+normB)^2+C*(HatC+2*normB+rhoS)))/(HatC+2*normB+rhoS);