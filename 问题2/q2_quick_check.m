function q2_quick_check()
%Q2_QUICK_CHECK  快速验证库函数与 Q1 基准（无需并行/优化）
thisDir = fileparts(mfilename('fullpath'));
addpath(fullfile(thisDir, 'lib'));

p = scenario_params();
x_q1 = [pi, 120, 1.5, 3.6];
[dur, info] = objective_q2(x_q1, p, 2e5);
[dur_r, tin, tout] = shield_duration(info.P_det, info.t_det, p, 2e5, true);

fprintf('Q1 复算: 粗=%.6f  细=%.6f  区间[%.6f, %.6f]\n', dur, dur_r, tin, tout);
fprintf('P_drop=[%.3f, %.3f, %.3f]\n', info.P_drop);
fprintf('P_det =[%.3f, %.3f, %.3f] t_det=%.3f\n', info.P_det, info.t_det);
assert(abs(dur_r - 1.4055) < 0.02, 'Q1 时长偏离过大');
fprintf('库函数自检通过。\n');
end
