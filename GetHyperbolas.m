function [d1,d2] = GetHyperbolas(kappa,B0,nEig)
%gives vectors of hyperbolas
y=zeros(nEig,1);
d1=zeros(500,nEig);
d2=zeros(500,nEig);
for N=1:nEig
    y(N)=B0(1,1)/kappa(N);
    d1(:,N)=linspace(0,(y(N)-y(N)/100),500);
    d2(:,N)=(B0(1,2)*B0(2,1)./(d1(:,N)*kappa(N)-B0(1,1)*ones(500,1))+B0(2,2)*ones(500,1))*(1/kappa(N));
end

