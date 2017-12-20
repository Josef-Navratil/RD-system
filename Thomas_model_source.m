clear all
L1=10;
L2=15;
tau=1e7;
b11=226.7;
b12=-1124.5;
b21=478.7;
b22=-1502.5-tau;
b220=-1502.5;
detB0=b11*b220-b21*b12;
detB=b11*b22-b12*b21;
TrB=b11+b22;

r=3.58;
size=1/(1+abs(b11*b220/detB0));
meas=r^2*pi/(L1*L2-r^2*pi);
Pomer=meas/size;
normB=tau;
sizeB=r^2*pi*tau;  %Characteristic function source
%sizeB=0.5*tau; %Errorfunction source
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
y3=b11/kappa21;
y4=b11/kappa22;
y5=b11/kappa13;
y6=b11/kappa31;
CM=((1+1/(eps0^2))/(1-kappa11))*rhoS;
d21=@(x) 1/kappa11*(b12*b21./(x.*kappa11-b11)+b22);
d22=@(x) 1/kappa12*(b12*b21./(x.*kappa12-b11)+b22);
d23=@(x) 1/kappa21*(b12*b21./(x.*kappa21-b11)+b22);
d24=@(x) 1/kappa13*(b12*b21./(x.*kappa22-b11)+b22);
d25=@(x) 1/kappa22*(b12*b21./(x.*kappa13-b11)+b22);
d26=@(x) 1/kappa31*(b12*b21./(x.*kappa31-b11)+b22);
xvec1=linspace(0,y1/3,500);
xvec2=linspace(0,y2/2,500);
xvec3=linspace(0,y3-50,500);
xvec4=linspace(0,y4-10,500);
xvec5=linspace(0,y5-10,500);
xvec6=linspace(0,y6-10,500);
tau0=b12*b21/(y2*kappa11-b11)+b220;
break
figure
plot(xvec1,d21(xvec1),xvec2,d22(xvec2),xvec3,d23(xvec3),xvec4,d24(xvec4),...
xvec5,d25(xvec5),xvec6,d26(xvec6)) 
title('Graph hyperbolas for Thomas model with sources')
xlabel('d_1') % x-axis label
ylabel('d_2') % y-axis label
legend('d_{2,1}','d_{2,2}','d_{2,3}','d_{2,4}','d_{2,5}','d_{2,6}','Location','southeast')
axis([0 400 0 20000])