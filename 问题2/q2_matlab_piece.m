function q2_matlab_piece(i0, i1, mode)
%Q2_MATLAB_PIECE  无并行池分块（扁平实现，避免嵌套函数）
if nargin < 3, mode = 'coarse'; end
if nargin < 2, i1 = 200; end
if nargin < 1, i0 = 1; end

thisDir = fileparts(mfilename('fullpath'));
addpath(fullfile(thisDir, 'lib'));
outDir = fullfile(thisDir, '结果');
if ~exist(outDir, 'dir'), mkdir(outDir); end

stateFile = fullfile(outDir, 'q2_matlab_state.mat');
logFile = fullfile(outDir, 'q2_matlab_serial.log');
tokenFile = fullfile(outDir, 'q2_last_token.txt');

p = scenario_params();
lb = [0; 70; 0; 0.3];
ub = [2*pi; 140; 25; 16];
nCoarse = 6000;
nRef = 4e4;
nHi = 2.5e5;

mode = lower(char(mode));
if strcmp(mode, 'init')
    lf = fopen(logFile, 'w');
    fprintf(lf, '==== MATLAB serial pieces, no parpool ====\n');
    fprintf('==== MATLAB serial pieces, no parpool ====\n');

    th = pi + linspace(-0.28, 0.28, 9);
    vv = [70 85 100 110 120 130 140];
    td = [0 0.2 0.5 1.0 1.5 2.0 2.5 3.5 5 7 10];
    ta = [1.5 2.0 2.5 3.0 3.5 4.0 5.0 6.5 8 10];
    nG = numel(th) * numel(vv) * numel(td) * numel(ta);
    Xg = zeros(nG, 4);
    k = 0;
    for a = 1:numel(th)
        for b = 1:numel(vv)
            for c = 1:numel(td)
                for d = 1:numel(ta)
                    k = k + 1;
                    Xg(k, :) = [th(a), vv(b), td(c), ta(d)];
                end
            end
        end
    end
    rng(2025);
    nR = 600;
    U = rand(nR, 4);
    Xr = zeros(nR, 4);
    for j = 1:4
        Xr(:, j) = lb(j) + U(:, j) .* (ub(j) - lb(j));
    end
    seeds = [ ...
        pi, 120, 1.5, 3.6; ...
        pi, 70, 0.0, 2.5; ...
        pi, 70, 0.2, 2.5; ...
        pi, 100, 2.0, 4.0; ...
        3.0881, 71.89, 0.0, 2.50; ...
        pi-0.05, 70, 0.0, 2.5; ...
        pi+0.05, 70, 0.0, 2.5];
    X = [seeds; Xg; Xr];
    nAll = size(X, 1);
    durs = -ones(nAll, 1);
    best = -1;
    bestX = X(1, :);
    nTop = 12;
    cX = zeros(nTop, 4);
    cXf = zeros(nTop, 4);
    cD = -ones(nTop, 1);
    cX_ready = false;
    done_coarse = false;
    done_refine = false;
    xbest = X(1, :);
    info = struct();
    dur_ref = NaN;
    tin = NaN;
    tout = NaN;
    d1 = 1.405510;

    save(stateFile, 'X', 'durs', 'nAll', 'best', 'bestX', 'd1', 'lb', 'ub', ...
        'nCoarse', 'nTop', 'cX', 'cXf', 'cD', 'cX_ready', 'done_coarse', ...
        'done_refine', 'xbest', 'info', 'dur_ref', 'tin', 'tout');

    fprintf(lf, 'Q1 = %.6f\n', d1);
    fprintf('Q1 = %.6f\n', d1);
    fprintf(lf, 'init N=%d nCoarse=%d\n', nAll, nCoarse);
    fprintf('init N=%d nCoarse=%d\n', nAll, nCoarse);
    fprintf(lf, 'INIT_OK\n');
    fprintf('INIT_OK\n');
    fclose(lf);
    fid = fopen(tokenFile, 'w');
    fprintf(fid, 'INIT_OK\n');
    fclose(fid);
    return;
end

if strcmp(mode, 'coarse')
    S = load(stateFile);
    i0 = max(1, round(i0));
    i1 = min(S.nAll, round(i1));
    lf = fopen(logFile, 'a');
    fprintf(lf, 'coarse [%d,%d]\n', i0, i1);
    fprintf('coarse [%d,%d]\n', i0, i1);
    t0 = tic;
    for i = i0:i1
        if S.durs(i) >= 0
            continue;
        end
        S.durs(i) = objective_q2(S.X(i, :), p, nCoarse);
        if S.durs(i) > S.best
            S.best = S.durs(i);
            S.bestX = S.X(i, :);
        end
    end
    save(stateFile, '-struct', 'S');
    fprintf(lf, 'coarse done [%d,%d] best=%.4f t=%.2f\n', i0, i1, S.best, toc(t0));
    fprintf('coarse done [%d,%d] best=%.4f th=%.2f v=%.1f td=%.2f tau=%.2f t=%.2f\n', ...
        i0, i1, S.best, mod(S.bestX(1) * 180 / pi, 360), S.bestX(2), S.bestX(3), S.bestX(4), toc(t0));
    fprintf(lf, 'PIECE_OK\n');
    fprintf('PIECE_OK\n');
    fclose(lf);
    fid = fopen(tokenFile, 'w');
    fprintf(fid, 'PIECE_OK\n');
    fclose(fid);
    return;
end

if strcmp(mode, 'refine')
    S = load(stateFile);
    if ~isfield(S, 'cX_ready') || ~S.cX_ready
        valid = S.durs >= 0;
        if ~all(valid)
            error('coarse incomplete: %d/%d', nnz(valid), S.nAll);
        end
        [ds, ord] = sort(S.durs, 'descend');
        Xs = S.X(ord, :);
        S.ds = ds;
        S.Xs = Xs;
        nt = min(S.nTop, S.nAll);
        S.cX = Xs(1:nt, :);
        S.cXf = S.cX;
        S.cD = -ones(nt, 1);
        S.cX_ready = true;
        S.done_coarse = true;
        save(stateFile, '-struct', 'S');
    end

    i0 = max(1, round(i0));
    i1 = min(size(S.cX, 1), round(i1));
    opts = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', ...
        'MaxFunctionEvaluations', 350, 'MaxIterations', 90, ...
        'OptimalityTolerance', 1e-8, 'StepTolerance', 1e-12, 'UseParallel', false);
    obj = @(x) -objective_q2(x, p, nRef);

    lf = fopen(logFile, 'a');
    fprintf(lf, 'refine [%d,%d]\n', i0, i1);
    fprintf('refine [%d,%d]\n', i0, i1);
    for k = i0:i1
        if S.cD(k) >= 0
            continue;
        end
        try
            [xf, fv] = fmincon(obj, S.cX(k, :), [], [], [], [], lb, ub, [], opts);
            S.cXf(k, :) = xf;
            S.cD(k) = -fv;
        catch ME
            S.cD(k) = objective_q2(S.cX(k, :), p, nRef);
            S.cXf(k, :) = S.cX(k, :);
            fprintf(lf, '  fmincon#%d fail %s\n', k, ME.message);
        end
        if isfield(S, 'ds')
            base = S.ds(k);
        else
            base = NaN;
        end
        fprintf(lf, '  refine %d: %.4f -> %.4f\n', k, base, S.cD(k));
        fprintf('  refine %d: %.4f -> %.4f\n', k, base, S.cD(k));
        save(stateFile, '-struct', 'S');
    end
    fprintf(lf, 'PIECE_OK\n');
    fprintf('PIECE_OK\n');
    fclose(lf);
    fid = fopen(tokenFile, 'w');
    fprintf(fid, 'PIECE_OK\n');
    fclose(fid);
    return;
end

if strcmp(mode, 'finalize')
    S = load(stateFile);
    if any(S.cD < 0)
        error('refine incomplete');
    end
    [bm, ib] = max(S.cD);
    S.xbest = S.cXf(ib, :);
    [~, info] = objective_q2(S.xbest, p, nHi);
    [dur_ref, tin, tout] = shield_duration(info.P_det, info.t_det, p, nHi, true);
    S.info = info;
    S.dur_ref = dur_ref;
    S.tin = tin;
    S.tout = tout;
    S.done_refine = true;
    save(stateFile, '-struct', 'S');

    lf = fopen(logFile, 'a');
    fprintf(lf, 'finalize best mid=%.6f hi=%.6f\n', bm, dur_ref);
    fprintf('finalize best mid=%.6f hi=%.6f\n', bm, dur_ref);

    outTxt = fullfile(outDir, 'q2_matlab_serial_result.txt');
    fid = fopen(outTxt, 'w');
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
    fprintf(fid, '问题1固定策略时长 = %.6f s\n', S.d1);
    fprintf(fid, '提升 Delta = %.6f s\n', dur_ref - S.d1);
    fprintf(fid, '粗搜样本数 = %d, 精修候选 = %d\n', S.nAll, size(S.cX, 1));
    fprintf(fid, '并行池 = 未使用\n');
    fprintf(fid, '求解方式 = 分块串行 MATLAB batch\n');
    fclose(fid);

    save(fullfile(outDir, 'q2_matlab_serial_result.mat'), '-struct', 'S');
    fid = fopen(fullfile(outDir, 'q2_matlab_serial_result.csv'), 'w');
    fprintf(fid, 'theta_rad,heading_deg,v_mps,t_drop_s,tau_s,t_det_s,Pdrop_x,Pdrop_y,Pdrop_z,Pdet_x,Pdet_y,Pdet_z,duration_s,t_in,t_out\n');
    fprintf(fid, '%.10f,%.6f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n', ...
        info.theta, info.heading_deg, info.v, info.t_drop, info.tau, info.t_det, ...
        info.P_drop(1), info.P_drop(2), info.P_drop(3), ...
        info.P_det(1), info.P_det(2), info.P_det(3), dur_ref, tin, tout);
    fclose(fid);

    fprintf(lf, 'DONE duration=%.6f\nFINAL_OK\n', dur_ref);
    fprintf('DONE duration=%.6f\nFINAL_OK\n', dur_ref);
    fclose(lf);
    fid = fopen(tokenFile, 'w');
    fprintf(fid, 'FINAL_OK\n');
    fclose(fid);
    return;
end

error('unknown mode %s', mode);
end
