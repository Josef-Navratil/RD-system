clear all
L1=10;  %Width of the domain
L2=15;  %Height of the domain
nEig=5; %Number of eigenvalues
tau=5e6;
BC='Neumann';
b11=226.7;
b12=-1124.5;
b21=478.7;
b22=-1502.5-tau;
b220=-1502.5;
B=[b11 b12; b21 b22];
B0=[b11 b12; b21 b220];
detB0=det(B0);
detB=det(B);
TrB=b11+b22;

if(strcmp(BC,'Neumann'))
rSource=3.57;
KineticSize=1/(1+abs(b11*b220/detB0));
RelativeSourceMeas=rSource^2*pi/(L1*L2-rSource^2*pi);
Ineq=RelativeSourceMeas/KineticSize;
normB=tau;
sizeB=rSource^2*pi*tau; %Characteristic function
%sizeB=0.5*tau; %Errorfunction source
HatC=7*(2*normB);
rhoS=detB0/b11;
C=-rhoS+1/(L1*L2)*sizeB;
tau0=(L1*L2*tau*detB0/b11)/(sizeB);
eps0=ComputeEps(HatC,rhoS,normB,C);
if(eps0<=0)
    disp('Too weak source')
end
end
kappa=GetEigenvalues(L1,L2,nEig,BC);
if(strcmp(BC,'Neumann'))
CM=(1+1/eps0^2)*rhoS/(1-1/(1+kappa(1)));
end
[d1,d2]=GetHyperbolas(kappa,B0,nEig);
colorVec = hsv(nEig);
figure
for N=1:nEig
   plot(d1(:,N),d2(:,N),'Color',colorVec(N,:));
   legendInfo{N} = sprintf('d_{2,%d}', N);
   hold on
end
legend(legendInfo,'Location','southeast')
title('Graph hyperbolas for Thomas model with sources')
xlabel('d_1') % x-axis label
ylabel('d_2') % y-axis label
axis([0 400 0 20000]) %set axis range
hold off
%clearvars -except detB detB0 Ineq eps0 CM