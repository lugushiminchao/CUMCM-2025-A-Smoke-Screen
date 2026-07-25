function q2_verify_best()
% 用 MATLAB 高精度复核 Python 最优解（短脚本，避免 batch 崩溃前写完）
thisDir = fileparts(mfilename('fullpath'));
addpath(fullfile(thisDir, 'lib'));
outDir = fullfile(thisDir, '结果');

p = scenario_params();

% 读取 Python 结果（若有），否则用已知最优
csvFile = fullfile(outDir, 'q2_result.csv');
if exist(csvFile, 'file')
    T = readmatrix(csvFile, 'NumHeaderLines', 1);
    x = T(1:4);  % theta, heading skip - csv has more cols
    % csv: theta,heading,v,tdrop,tau,...
    theta = T(1); v = T(3); t_drop = T(4); tau = T(5);
else
    theta = 3.0880757565; v = 71.8890217683; t_drop = 0; tau = 2.5032397513;
end
x = [theta, v, t_drop, tau];

[dur, info] = objective_q2(x, p, 3e5);
[dur_r, tin, tout] = shield_duration(info.P_det, info.t_det, p, 4e5, true);

% Q1
xq1 = [pi,120,1.5,3.6];
[~, i1] = objective_q2(xq1, p, 1e5);
[d1,~,~] = shield_duration(i1.P_det, i1.t_det, p, 2e5, true);

fprintf('MATLAB 复核最优:\n');
fprintf('  dur_coarse=%.6f  refine=%.6f  [% .6f, %.6f]\n', dur, dur_r, tin, tout);
fprintf('  theta=%.10f  v=%.10f  td=%.10f  tau=%.10f\n', theta, v, t_drop, tau);
fprintf('  P_drop=[%.6f %.6f %.6f]\n', info.P_drop);
fprintf('  P_det =[%.6f %.6f %.6f] t_det=%.6f\n', info.P_det, info.t_det);
fprintf('  Q1 refine=%.6f  gain=%.6f\n', d1, dur_r-d1);

fid = fopen(fullfile(outDir, 'q2_matlab_verify.txt'), 'w', 'n', 'UTF-8');
fprintf(fid, 'MATLAB 高精度复核 Python 最优解\n');
fprintf(fid, '有效遮蔽时长 = %.6f s\n', dur_r);
fprintf(fid, '有效区间 = [%.6f, %.6f] s\n', tin, tout);
fprintf(fid, 'theta=%.10f rad, v=%.10f, t_drop=%.10f, tau=%.10f\n', theta,v,t_drop,tau);
fprintf(fid, 'P_drop=[%.10f, %.10f, %.10f]\n', info.P_drop);
fprintf(fid, 'P_det =[%.10f, %.10f, %.10f]\n', info.P_det);
fprintf(fid, 't_det=%.10f\n', info.t_det);
fprintf(fid, 'Q1=%.6f, Delta=%.6f\n', d1, dur_r-d1);
fclose(fid);
fprintf('已写入 q2_matlab_verify.txt\n');
end
