clc;
clear;
M1 = zeros(51,30);
% ll = [10,30,50,100];
ll = [10,30,50,100];
M2 = zeros(51,30);
M3 = zeros(51,30);
M4 = zeros(51,30);
for j = 1:4
for i = 1:30
%     eval(['a = dlmread(CoDE_' num2str(i) '_' num2str(ll(j)) ')']);
      exp = ['EBOwithCMAR_' num2str(i) '_' num2str(ll(j)) '.dat'];
      a = dlmread(exp);
    a(1:13,:)=[];a = a';
    eval(['M' num2str(j) '(:,' num2str(i) ')=a;' ]);
end
end
B1 = mean(M1,1)';
B2 = mean(M2,1)';
B3 = mean(M3,1)';
B4 = mean(M4,1)';
B = [B1,B2,B3,B4];
C = sum(B);