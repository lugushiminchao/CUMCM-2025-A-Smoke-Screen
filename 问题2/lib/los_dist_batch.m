function [d, s] = los_dist_batch(ts, M0, v_m, P_det, t_det, v_sink, Tgt)
%LOS_DIST_BATCH  云团中心到导弹-目标视线段的距离（向量化）
%   [d,s] = los_dist_batch(ts, M0, v_m, P_det, t_det, v_sink, Tgt)
%   d : 点到线段最短距离
%   s : 视线参数（0=导弹，1=目标；落在[0,1]表示投影在线段上）

ts = ts(:);
Mx = M0(1) + v_m(1)*ts;
My = M0(2) + v_m(2)*ts;
Mz = M0(3) + v_m(3)*ts;

Cx = P_det(1) + zeros(size(ts));
Cy = P_det(2) + zeros(size(ts));
Cz = P_det(3) - v_sink*(ts - t_det);

ABx = Tgt(1) - Mx; ABy = Tgt(2) - My; ABz = Tgt(3) - Mz;
ACx = Cx - Mx;     ACy = Cy - My;     ACz = Cz - Mz;

L2 = ABx.^2 + ABy.^2 + ABz.^2;
s  = (ACx.*ABx + ACy.*ABy + ACz.*ABz) ./ max(L2, 1e-18);
sc = min(max(s, 0), 1);

px = Mx + sc.*ABx; py = My + sc.*ABy; pz = Mz + sc.*ABz;
d  = sqrt((Cx-px).^2 + (Cy-py).^2 + (Cz-pz).^2);
end
