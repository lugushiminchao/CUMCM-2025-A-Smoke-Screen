function q2_smoke()
% 最小冒烟：确认循环求值不会崩
thisDir = fileparts(mfilename('fullpath'));
addpath(fullfile(thisDir, 'lib'));
p = scenario_params();
fprintf('start smoke\n');
X = [
    pi, 120, 1.5, 3.6;
    pi, 140, 0.5, 3.0;
    pi, 100, 2.0, 4.0;
    pi+0.1, 110, 1.0, 4.0;
    pi-0.1, 130, 0.8, 2.5;
    ];
for i = 1:size(X,1)
    d = objective_q2(X(i,:), p, 5000);
    fprintf('i=%d d=%.4f\n', i, d);
end
% 稍大一点
n = 200;
rng(1);
U = rand(n,4);
lb = [0,70,0,0.3]; ub = [2*pi,140,25,16];
for i = 1:n
    x = lb + U(i,:).*(ub-lb);
    d = objective_q2(x, p, 8000);
    if mod(i,40)==0
        fprintf('rand %d ok last=%.4f\n', i, d);
    end
end
fprintf('smoke done\n');
end
