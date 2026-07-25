function [dur, t_in, t_out] = shield_duration(P_det, t_det, p, n, refine)
%SHIELD_DURATION  单枚烟幕对 M1 的有效遮蔽时长（可含多段）
%   [dur,t_in,t_out] = shield_duration(P_det, t_det, p, n)
%   [dur,t_in,t_out] = shield_duration(..., true)  % 二分细化边界后重算

if nargin < 4 || isempty(n), n = 5e4; end
if nargin < 5, refine = false; end

t0 = t_det;
t1 = min(t_det + p.T_life, p.t_hit);
if ~(t1 > t0)
    dur = 0; t_in = NaN; t_out = NaN;
    return;
end

ts = linspace(t0, t1, n);
[d, s] = los_dist_batch(ts, p.M0, p.v_m, P_det, t_det, p.v_sink, p.Tgt);
ok = (d <= p.R) & (s >= 0) & (s <= 1);
dt = ts(2) - ts(1);
dur = sum(ok) * dt;

if ~any(ok)
    t_in = NaN; t_out = NaN;
    return;
end

jump = find(diff(ok) ~= 0);
if isempty(jump)
    % 全程有效
    t_in = t0; t_out = t1;
    if refine, dur = t_out - t_in; end
    return;
end

% 细化每个跳变边界
edges = zeros(size(jump));
for i = 1:numel(jump)
    j = jump(i);
    lo = ts(j); hi = ts(j+1);
    left_ok = ok(j);
    if refine
        for it = 1:40
            mid = 0.5*(lo + hi);
            [dm, sm] = los_dist_batch(mid, p.M0, p.v_m, P_det, t_det, p.v_sink, p.Tgt);
            mid_ok = (dm <= p.R) && (sm >= 0) && (sm <= 1);
            if mid_ok == left_ok
                lo = mid;
            else
                hi = mid;
            end
        end
    end
    edges(i) = 0.5*(lo + hi);
end

% 由边序列重建各段，累加时长
segs = zeros(0, 2);
state = ok(1);
cur_in = t0;
for i = 1:numel(edges)
    if state
        segs(end+1, :) = [cur_in, edges(i)]; %#ok<AGROW>
        state = false;
    else
        cur_in = edges(i);
        state = true;
    end
end
if state
    segs(end+1, :) = [cur_in, t1]; %#ok<AGROW>
end

if isempty(segs)
    dur = 0; t_in = NaN; t_out = NaN;
else
    if refine
        dur = sum(segs(:,2) - segs(:,1));
    end
    t_in  = segs(1,1);
    t_out = segs(1,2);
end
end
