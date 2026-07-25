function q2_matlab_chunk()
% 分块串行：粗搜→落盘→精修→落盘（无并行池）
% 结果写在 结果/q2_matlab_serial_*

thisDir = fileparts(mfilename('fullpath'));
addpath(fullfile(thisDir, 'lib'));
outDir = fullfile(thisDir, '结果');
if ~exist(outDir, 'dir'), mkdir(outDir); end

logPath = fullfile(outDir, 'q2_matlab_serial.log');
lf = fopen(logPath, 'w', 'n', 'UTF-8');
tAll = tic;
p = scenario_params();

fprintf(lf, '==== MATLAB serial no parpool ====\n');
fprintf('==== MATLAB serial no parpool ====\n');

% Q1
xq1 = [pi, 120, 1.5, 3.6];
[~, i1] = objective_q2(xq1, p, 5e4);
[d1, ~, ~] = shield_duration(i1.P_det, i1.t_det, p, 1e5, true);
fprintf(lf, 'Q1 = %.6f\n', d1); fprintf('Q1 = %.6f\n', d1);

lb = [0; 70; 0; 0.3];
ub = [2*pi; 140; 25; 16];
nCoarse = 6000;

% 较小网格加快
th = pi + linspace(-0.28, 0.28, 9);
vv = [70 85 100 110 120 130 140];
td = [0 0.2 0.5 1.0 1.5 2.0 2.5 3.5 5 7 10];
ta = [1.5 2.0 2.5 3.0 3.5 4.0 5.0 6.5 8 10];
nG = numel(th)*numel(vv)*numel(td)*numel(ta);
Xg = zeros(nG, 4);
k = 0;
for a = 1:numel(th)
    for b = 1:numel(vv)
        for c = 1:numel(td)
            for d = 1:numel(ta)
                k = k + 1;
                Xg(k,:) = [th(a), vv(b), td(c), ta(d)];
            end
        end
    end
end

rng(2025);
nR = 600;
Xr = zeros(nR, 4);
U = rand(nR, 4);
for j = 1:4
    Xr(:,j) = lb(j) + U(:,j).*(ub(j) - lb(j));
end
seeds = [
    pi, 120, 1.5, 3.6;
    pi, 70, 0.0, 2.5;
    pi, 70, 0.2, 2.5;
    pi, 100, 2.0, 4.0;
    3.0881, 71.89, 0.0, 2.50;
    pi-0.05, 70, 0.0, 2.5;
    pi+0.05, 70, 0.0, 2.5;
    ];
X = [seeds; Xg; Xr];
nAll = size(X, 1);
durs = zeros(nAll, 1);
fprintf(lf, 'N = %d\n', nAll); fprintf('N = %d\n', nAll);

best = -1; bestX = X(1,:);
tC = tic;
chunk = 500;
for i = 1:nAll
    durs(i) = objective_q2(X(i,:), p, nCoarse);
    if durs(i) > best
        best = durs(i);
        bestX = X(i,:);
    end
    if mod(i, chunk) == 0 || i == nAll
        fprintf(lf, 'i=%d/%d best=%.4f th=%.2f v=%.1f td=%.2f tau=%.2f t=%.1f\n', ...
            i, nAll, best, mod(bestX(1)*180/pi,360), bestX(2), bestX(3), bestX(4), toc(tC));
        fprintf('i=%d/%d best=%.4f th=%.2f v=%.1f td=%.2f tau=%.2f t=%.1f\n', ...
            i, nAll, best, mod(bestX(1)*180/pi,360), bestX(2), bestX(3), bestX(4), toc(tC));
        save(fullfile(outDir, 'q2_matlab_progress.mat'), 'best', 'bestX', 'i', 'durs', 'X', 'nAll');
    end
end

[ds, ord] = sort(durs, 'descend');
Xs = X(ord, :);
save(fullfile(outDir, 'q2_matlab_coarse.mat'), 'Xs', 'ds', 'bestX', 'best', 'd1');
fprintf(lf, 'coarse done best=%.4f\n', best);
fprintf('coarse done best=%.4f\n', best);
for k = 1:min(8, nAll)
    fprintf('  #%d %.4f th=%.2f v=%.1f td=%.3f tau=%.3f\n', ...
        k, ds(k), mod(Xs(k,1)*180/pi,360), Xs(k,2), Xs(k,3), Xs(k,4));
    fprintf(lf, '  #%d %.4f th=%.2f v=%.1f td=%.3f tau=%.3f\n', ...
        k, ds(k), mod(Xs(k,1)*180/pi,360), Xs(k,2), Xs(k,3), Xs(k,4));
end

% fmincon
nTop = 12;
nRef = 4e4;
opts = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', ...
    'MaxFunctionEvaluations', 400, 'MaxIterations', 100, ...
    'OptimalityTolerance', 1e-8, 'StepTolerance', 1e-12, 'UseParallel', false);
obj = @(x) -objective_q2(x, p, nRef);
cX = Xs(1:min(nTop, nAll), :);
cXf = cX; cD = zeros(size(cX,1), 1);
for k = 1:size(cX, 1)
    try
        [xf, fv] = fmincon(obj, cX(k,:), [], [], [], [], lb, ub, [], opts);
        cXf(k,:) = xf; cD(k) = -fv;
    catch
        cD(k) = objective_q2(cX(k,:), p, nRef);
        cXf(k,:) = cX(k,:);
    end
    fprintf('refine %d/%d: %.4f -> %.4f\n', k, size(cX,1), ds(k), cD(k));
    fprintf(lf, 'refine %d/%d: %.4f -> %.4f\n', k, size(cX,1), ds(k), cD(k));
    save(fullfile(outDir, 'q2_matlab_refine_progress.mat'), 'cXf', 'cD', 'k');
end

[bm, ib] = max(cD);
xbest = cXf(ib, :);
fprintf('mid best = %.6f\n', bm);
fprintf(lf, 'mid best = %.6f\n', bm);

% highres
nHi = 2.5e5;
[~, info] = objective_q2(xbest, p, nHi);
[dur_ref, tin, tout] = shield_duration(info.P_det, info.t_det, p, nHi, true);

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
fprintf(fid, '问题1固定策略时长 = %.6f s\n', d1);
fprintf(fid, '提升 Delta = %.6f s\n', dur_ref - d1);
fprintf(fid, '粗搜样本数 = %d, 精修候选 = %d\n', nAll, size(cX,1));
fprintf(fid, '总运行时间 = %.2f s\n', toc(tAll));
fprintf(fid, '并行池 = 未使用\n');
fclose(fid);

save(fullfile(outDir, 'q2_matlab_serial_result.mat'), ...
    'xbest', 'info', 'dur_ref', 'tin', 'tout', 'd1', 'cXf', 'cD', 'Xs', 'ds', 'p');

fid = fopen(fullfile(outDir, 'q2_matlab_serial_result.csv'), 'w', 'n', 'UTF-8');
fprintf(fid, 'theta_rad,heading_deg,v_mps,t_drop_s,tau_s,t_det_s,Pdrop_x,Pdrop_y,Pdrop_z,Pdet_x,Pdet_y,Pdet_z,duration_s,t_in,t_out\n');
fprintf(fid, '%.10f,%.6f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n', ...
    info.theta, info.heading_deg, info.v, info.t_drop, info.tau, info.t_det, ...
    info.P_drop(1), info.P_drop(2), info.P_drop(3), ...
    info.P_det(1), info.P_det(2), info.P_det(3), dur_ref, tin, tout);
fclose(fid);

fprintf(lf, 'DONE duration=%.6f total=%.1fs\n', dur_ref, toc(tAll));
fprintf('DONE duration=%.6f total=%.1fs\n', dur_ref, toc(tAll));
fclose(lf);
end
