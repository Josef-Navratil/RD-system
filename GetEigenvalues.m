function [ kappa ] = GetEigenvalues(L1,L2,nEig,BC)
%Gives eigenvalues of Laplacian on rectangle domain
kappa=zeros(nEig,nEig);
for m=1:nEig
    for n=1:nEig
        if(strcmp(BC,'Dirichlet'))
           kappa(m,n)=pi^2*(m^2/((L1)^2) + n^2/((L2)^2));
        end
        if(strcmp(BC,'Neumann'))
            kappa(m,n)=pi^2*((m-1)^2/((L1)^2) + (n-1)^2/((L2)^2));
        end
    end
end
kappa=sort(kappa(:),'ascend');
kappa=kappa(2:nEig+1);

