clear all
L1=10;
tau=0.2;
L2=15;
a=0.1;
b=0.5;
b11=-1+2*b/(a+b);
b12=(a+b)^2;
b21=-2*b/(a+b);
b220=-(a+b)^2;
b22S=-(a+b)^2-tau;
detB=b11*b220-b12*b21;
TrB=b11+b220;

detB0=b11*(b220+tau*0.865)-b21*b12;
normB=0.865*tau;
sizeB=4*pi*0.865*tau;  %Characteristic function source
%sizeB=0.1656*tau; %Errorfunction source
C0=detB0*(L1*L2*tau)/(b11*sizeB);
rhoS=detB0/b11;
C=-detB0/b11 + sizeB/(L1*L2);
hatC=7*(2*normB);
eps0=(-(rhoS+normB)+sqrt((rhoS+normB)^2+C*(hatC+2*normB+rhoS)))/(hatC+2*normB+rhoS);
f=@(x,y) (pi/L1)^2.*(x.^2) + (pi/L2)^2.*(y.^2);
kappa11=f(1,1);
kappa12=f(1,2);
kappa21=f(2,1);
kappa22=f(2,2);
kappa31=f(3,1);
kappa13=f(1,3);
y1=b11/kappa11;
y2=b11/kappa12;
y3=b11/kappa13;
y4=b11/kappa21;
y5=b11/kappa22;
y6=b11/kappa31;
CM=((1+1/(eps0^2))/(1-kappa11))*rhoS;

d21=@(x) 1/kappa11*(b12*b21./(x.*kappa11-b11)+b220);
d22=@(x) 1/kappa12*(b12*b21./(x.*kappa12-b11)+b220);
d23=@(x) 1/kappa13*(b12*b21./(x.*kappa13-b11)+b220);
%d23=@(x) 0;
d24=@(x) 1/kappa21*(b12*b21./(x.*kappa21-b11)+b220);
d25=@(x) 1/kappa22*(b12*b21./(x.*kappa22-b11)+b220);
d26=@(x) 1/kappa31*(b12*b21./(x.*kappa31-b11)+b220);

d21S=@(x) 1/kappa11*(b12*b21./(x.*kappa11-b11)+b22S);
d22S=@(x) 1/kappa12*(b12*b21./(x.*kappa12-b11)+b22S);
d23S=@(x) 1/kappa13*(b12*b21./(x.*kappa13-b11)+b22S);
%d23S=@(x) 0;
d24S=@(x) 1/kappa21*(b12*b21./(x.*kappa21-b11)+b22S);
d25S=@(x) 1/kappa22*(b12*b21./(x.*kappa22-b11)+b22S);
d26S=@(x) 1/kappa31*(b12*b21./(x.*kappa31-b11)+b22S);
xvec1=linspace(0,9/10*y1,500);
xvec2=linspace(0,9/10*y2,500);
xvec3=linspace(0,9/10*y3,500);
xvec4=linspace(0,9/10*y4,500);
xvec5=linspace(0,9/10*y5,500);
xvec6=linspace(0,(9/10)*y6,500);
d21(y2)

figure
plot(xvec1,d21(xvec1),'b',xvec2,d22(xvec2),'g',xvec3,d23(xvec3),'y',xvec4,d24(xvec4),'r',...
xvec5,d25(xvec5),'k',xvec6,d26(xvec6),'m',xvec1,d21S(xvec1),':b',xvec2,d22S(xvec2),':g',xvec3,d23S(xvec3),...
':y',xvec4,d24S(xvec4),':r',xvec5,d25S(xvec5),':k',xvec6,d26S(xvec6),':m') 
title('Graph of hyperbolas for Schnackenberg model with Dirichlet b.c.')
xlabel('d_1') % x-axis label
ylabel('d_2') % y-axis label
legend('d_{2,1}','d_{2,2}','d_{2,3}','d_{2,4}','d_{2,5}','d_{2,6}','Location','southeast')
axis([0 2 0 20])