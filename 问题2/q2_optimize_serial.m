function q2_optimize_serial()
%Q2_OPTIMIZE_SERIAL  问题2：MATLAB 串行优化（不启动并行池）
% 决策变量 x = [theta, v, t_drop, tau]
% 用法（在问题2目录，或从仓库根目录用 -sd 指定本目录）:
%   matlab -batch "q2_optimize_serial"
%   matlab -sd ".\问题2" -batch "q2_optimize_serial"

tAll = tic;
thisDir = fileparts(mfilename('fullpath'));
addpath(fullfile(thisDir, 'lib'));
outDir = fullfile(thisDir, '结果');
if ~exist(outDir, 'dir'), mkdir(outDir); end

logPath = fullfile(outDir, 'q2_matlab_serial.log');
lf = fopen(logPath, 'w', 'n', 'UTF-8');

p = scenario_params();
say(lf, '==== 问题2 MATLAB 串行优化（无并行池）====\n');
say(lf, sprintf('t_hit=%.6f s\n', p.t_hit));

% 确认未开并行池
pool = gcp('nocreate');
if ~isempty(pool)
    say(lf, sprintf('检测到并行池 %d workers，将删除以避免干扰\n', pool.NumWorkers));
    delete(pool);
end
say(lf, '并行池: 关闭\n');

%% Q1 基准
x_q1 = [pi, 120, 1.5, 3.6];
[~, info_q1] = objective_q2(x_q1, p, 8e4);
[dur_q1r, tin1, tout1] = shield_duration(info_q1.P_det, info_q1.t_det, p, 1.5e5, true);
say(lf, sprintf('[Q1] %.6f s  [% .6f, %.6f]\n', dur_q1r, tin1, tout1));

lb = [0; 70; 0.0; 0.3];
ub = [2*pi; 140; 25; 16];
nCoarse = 8000;

%% 结构化网格（预分配，规模适中）
th_list  = pi + linspace(-0.30, 0.30, 11);
v_list   = [70, 85, 100, 110, 120, 130, 140];
td_list  = [0.0, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.5, 5.0, 7.0, 10.0];
tau_list = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.5, 6.5, 8.0, 10.0];

nG = numel(th_list)*numel(v_list)*numel(td_list)*numel(tau_list);
Xg = zeros(nG, 4);
ig = 0;
for a = 1:numel(th_list)
    for b = 1:numel(v_list)
        for c = 1:numel(td_list)
            for d = 1:numel(tau_list)
                ig = ig + 1;
                Xg(ig,:) = [th_list(a), v_list(b), td_list(c), tau_list(d)];
            end
        end
    end
end

rng(2025, 'twister');
nR = 1000;
Ur = rand(nR, 4);
Xr = zeros(nR, 4);
for j = 1:4
    Xr(:,j) = lb(j) + Ur(:,j).*(ub(j)-lb(j));
end
nB = 500;
Xb = [mod(pi + (rand(nB,1)-0.5)*0.45, 2*pi), ...
      70 + rand(nB,1)*70, ...
      0.0 + rand(nB,1)*10, ...
      0.8 + rand(nB,1)*9];

seeds = [
    pi, 120, 1.5, 3.6;
    pi, 100, 2.0, 4.0;
    pi, 140, 0.5, 3.0;
    pi,  70, 0.2, 2.5;
    pi,  70, 0.0, 2.5;
    pi,  90, 2.0, 5.0;
    pi, 110, 1.5, 4.0;
    pi+0.05, 70, 0.2, 2.5;
    pi-0.05, 70, 0.2, 2.5;
    3.0881, 71.89, 0.0, 2.50;  % Python 最优附近种子
    ];

X = [seeds; Xg; Xr; Xb];
nAll = size(X, 1);
durs = zeros(nAll, 1);
say(lf, sprintf('粗搜样本 %d, n_time=%d（串行）\n', nAll, nCoarse));

best = -1;
bestX = X(1,:);
re = max(1, floor(nAll/20));
tC = tic;
for i = 1:nAll
    durs(i) = objective_q2(X(i,:), p, nCoarse);
    if durs(i) > best
        best = durs(i);
        bestX = X(i,:);
    end
    if mod(i, re) == 0 || i == nAll
        say(lf, sprintf('  %5d/%d  best=%.4f  th=%.2f v=%.1f td=%.2f tau=%.2f  (%.1fs)\n', ...
            i, nAll, best, mod(bestX(1)*180/pi,360), bestX(2), bestX(3), bestX(4), toc(tC)));
        % 中途落盘
        save(fullfile(outDir, 'q2_matlab_progress.mat'), 'best', 'bestX', 'i', 'durs', 'X');
    end
end
say(lf, sprintf('粗搜完成 %.1f s, best=%.4f\n', toc(tC), best));

[ds, ord] = sort(durs, 'descend');
Xs = X(ord, :);
say(lf, 'Top-10 粗搜:\n');
for k = 1:min(10, nAll)
    xk = Xs(k,:);
    say(lf, sprintf('  #%2d %.4f  th=%.2f v=%.1f td=%.3f tau=%.3f\n', ...
        k, ds(k), mod(xk(1)*180/pi,360), xk(2), xk(3), xk(4)));
end
save(fullfile(outDir, 'q2_matlab_coarse.mat'), 'Xs', 'ds', 'bestX', 'best', 'dur_q1r');

%% fmincon 精修（串行，UseParallel=false）
nTop = 15;
nRef = 5e4;
say(lf, sprintf('\n---- fmincon 精修 Top-%d（UseParallel=false）----\n', nTop));
opts = optimoptions('fmincon', ...
    'Display', 'off', ...
    'Algorithm', 'sqp', ...
    'MaxFunctionEvaluations', 450, ...
    'MaxIterations', 120, ...
    'OptimalityTolerance', 1e-8, ...
    'StepTolerance', 1e-12, ...
    'UseParallel', false);
obj = @(x) -objective_q2(x, p, nRef);

cX = Xs(1:min(nTop, nAll), :);
cXf = cX;
cD = zeros(size(cX,1), 1);
tR = tic;
for k = 1:size(cX,1)
    try
        [xf, fv] = fmincon(obj, cX(k,:), [], [], [], [], lb, ub, [], opts);
        cXf(k,:) = xf;
        cD(k) = -fv;
    catch ME
        say(lf, sprintf('  fmincon#%d 失败: %s\n', k, ME.message));
        cD(k) = objective_q2(cX(k,:), p, nRef);
        cXf(k,:) = cX(k,:);
    end
    say(lf, sprintf('  %2d/%d: %.4f -> %.4f\n', k, size(cX,1), ds(k), cD(k)));
end
say(lf, sprintf('精修完成 %.1f s\n', toc(tR)));

[bm, ib] = max(cD);
xbest = cXf(ib, :);
say(lf, sprintf('精修最优(中精度) = %.6f\n', bm));

%% 可选 patternsearch（关闭并行）
try
    ops = optimoptions('patternsearch', ...
        'Display', 'off', ...
        'MaxFunctionEvaluations', 600, ...
        'UseCompletePoll', true, ...
        'UseParallel', false, ...
        'MeshTolerance', 1e-9);
    [xps, ~] = patternsearch(@(x) -objective_q2(x, p, 5e4), xbest, ...
        [], [], [], [], lb, ub, [], ops);
    dps = objective_q2(xps, p, 6e4);
    say(lf, sprintf('patternsearch = %.6f\n', dps));
    if dps > bm + 1e-4
        xbest = xps;
        bm = dps;
        say(lf, '采用 patternsearch 解\n');
    end
catch ME
    say(lf, sprintf('patternsearch 跳过: %s\n', ME.message));
end

%% 高精度复核
nHi = 3e5;
[~, info] = objective_q2(xbest, p, nHi);
[dur_ref, tin, tout] = shield_duration(info.P_det, info.t_det, p, nHi, true);
info.dur = dur_ref;
info.t_in = tin;
info.t_out = tout;

say(lf, sprintf('\n==== MATLAB 串行最优 ====\n'));
say(lf, sprintf('有效遮蔽时长 = %.6f s\n', dur_ref));
say(lf, sprintf('有效区间 = [%.6f, %.6f] s\n', tin, tout));
say(lf, sprintf('theta=%.10f rad (%.4f deg)\n', info.theta, info.heading_deg));
say(lf, sprintf('v=%.10f m/s\n', info.v));
say(lf, sprintf('t_drop=%.10f s\n', info.t_drop));
say(lf, sprintf('tau=%.10f s\n', info.tau));
say(lf, sprintf('t_det=%.10f s\n', info.t_det));
say(lf, sprintf('P_drop=[%.10f, %.10f, %.10f]\n', info.P_drop));
say(lf, sprintf('P_det =[%.10f, %.10f, %.10f]\n', info.P_det));
say(lf, sprintf('Q1 %.6f -> %.6f (Delta=%.6f)\n', dur_q1r, dur_ref, dur_ref-dur_q1r));

%% 写正式结果（退出前必须完成）
outTxt = fullfile(outDir, 'q2_matlab_serial_result.txt');
fid = fopen(outTxt, 'w', 'n', 'UTF-8');
fprintf(fid, '==== 2025国赛A题 问题2 结果（MATLAB 串行，无并行池）====\n');
fprintf(fid, '判据: 云团球心到导弹-真目标中心视线段距离 <= 10 m\n');
fprintf(fid, '有效遮蔽时长 = %.6f s\n', dur_ref);
fprintf(fid, '有效区间(首段) = [%.6f, %.6f] s\n', tin, tout);
fprintf(fid, '\n--- 决策变量 ---\n');
fprintf(fid, '航向角 theta = %.10f rad = %.6f deg\n', info.theta, info.heading_deg);
fprintf(fid, '飞行速度 v   = %.10f m/s\n', info.v);
fprintf(fid, '投放时刻 t_drop = %.10f s\n', info.t_drop);
fprintf(fid, '引信延时 tau    = %.10f s\n', info.tau);
fprintf(fid, '\n--- 导出量 ---\n');
fprintf(fid, '速度向量 v_fy = [%.10f, %.10f, %.10f] m/s\n', info.v_fy);
fprintf(fid, '投放点 P_drop = [%.10f, %.10f, %.10f] m\n', info.P_drop);
fprintf(fid, '起爆时刻 t_det = %.10f s\n', info.t_det);
fprintf(fid, '起爆点 P_det  = [%.10f, %.10f, %.10f] m\n', info.P_det);
fprintf(fid, '\n--- 对照 ---\n');
fprintf(fid, '问题1固定策略时长 = %.6f s\n', dur_q1r);
fprintf(fid, '提升 Delta = %.6f s\n', dur_ref - dur_q1r);
fprintf(fid, '粗搜样本数 = %d, 精修候选 = %d\n', nAll, size(cX,1));
fprintf(fid, '总运行时间 = %.2f s\n', toc(tAll));
fprintf(fid, '并行池 = 未使用\n');
fclose(fid);

save(fullfile(outDir, 'q2_matlab_serial_result.mat'), ...
    'xbest', 'info', 'dur_ref', 'tin', 'tout', 'dur_q1r', 'cXf', 'cD', 'Xs', 'ds', 'p');

fid = fopen(fullfile(outDir, 'q2_matlab_serial_result.csv'), 'w', 'n', 'UTF-8');
fprintf(fid, 'theta_rad,heading_deg,v_mps,t_drop_s,tau_s,t_det_s,Pdrop_x,Pdrop_y,Pdrop_z,Pdet_x,Pdet_y,Pdet_z,duration_s,t_in,t_out\n');
fprintf(fid, '%.10f,%.6f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n', ...
    info.theta, info.heading_deg, info.v, info.t_drop, info.tau, info.t_det, ...
    info.P_drop(1), info.P_drop(2), info.P_drop(3), ...
    info.P_det(1), info.P_det(2), info.P_det(3), dur_ref, tin, tout);
fclose(fid);

say(lf, sprintf('\n结果已写入:\n  %s\n', outTxt));
say(lf, sprintf('全部完成，总耗时 %.2f s\nDONE\n', toc(tAll)));
fclose(lf);
end

function say(lf, msg)
fprintf('%s', msg);
if lf > 0
    fprintf(lf, '%s', msg);
end
end
