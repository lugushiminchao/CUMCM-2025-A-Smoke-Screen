function q2_optimize()
%Q2_OPTIMIZE  问题2优化（结果随时落盘；兼容 batch 退出崩溃）
thisDir = fileparts(mfilename('fullpath'));
addpath(fullfile(thisDir, 'lib'));
outDir = fullfile(thisDir, '结果');
if ~exist(outDir, 'dir'), mkdir(outDir); end
progFile = fullfile(outDir, 'q2_progress.txt');
logfid = fopen(progFile, 'w', 'n', 'UTF-8');

tAll = tic;
p = scenario_params();
plog(logfid, '==== 问题2 优化开始 ====\n');

% Q1
x_q1 = [pi, 120, 1.5, 3.6];
[~, iq1] = objective_q2(x_q1, p, 8e4);
[dur_q1r, tin1, tout1] = shield_duration(iq1.P_det, iq1.t_det, p, 1.5e5, true);
plog(logfid, sprintf('[Q1] %.6f s [% .6f, %.6f]\n', dur_q1r, tin1, tout1));

lb = [0; 70; 0; 0.3];
ub = [2*pi; 140; 25; 16];
nCoarse = 8000;

% 紧凑网格（约 8k）
th_list  = pi + linspace(-0.28, 0.28, 11);
v_list   = [70 90 100 110 120 130 140];
td_list  = [0.3 0.6 1.0 1.5 2.0 2.5 3.5 5.0 7.0 10];
tau_list = [1.5 2.5 3.0 3.5 4.0 4.5 5.5 6.5 8.0 10];
nG = numel(th_list)*numel(v_list)*numel(td_list)*numel(tau_list);
Xg = zeros(nG,4); ig=0;
for a=1:numel(th_list)
  for b=1:numel(v_list)
    for c=1:numel(td_list)
      for d=1:numel(tau_list)
        ig=ig+1; Xg(ig,:)=[th_list(a),v_list(b),td_list(c),tau_list(d)];
      end
    end
  end
end

rng(2025);
nR=800; U=rand(nR,4); Xr=zeros(nR,4);
for j=1:4, Xr(:,j)=lb(j)+U(:,j).*(ub(j)-lb(j)); end
nB=400;
Xb=[mod(pi+(rand(nB,1)-0.5)*0.4,2*pi), 70+rand(nB,1)*70, 0.2+rand(nB,1)*10, 1+rand(nB,1)*9];
seeds=[pi,120,1.5,3.6; pi,100,2.0,4.0; pi,140,0.5,3.0; pi,90,2.0,5.0; ...
       pi,110,1.5,4.0; pi,100,1.5,4.5; pi,130,2.5,3.5; pi+0.08,100,2.0,4.0; ...
       pi-0.08,100,2.0,4.0; pi,70,1.0,5.0; pi,140,2.0,3.5; pi,120,3.0,4.0];
X=[seeds;Xg;Xr;Xb];
nAll=size(X,1);
durs=zeros(nAll,1);
plog(logfid, sprintf('粗搜样本 %d, n_time=%d\n', nAll, nCoarse));

best=0; bestX=X(1,:);
re=max(1,floor(nAll/25));
tC=tic;
for i=1:nAll
    durs(i)=objective_q2(X(i,:),p,nCoarse);
    if durs(i)>best
        best=durs(i); bestX=X(i,:);
    end
    if mod(i,re)==0 || i==nAll
        msg=sprintf('  %d/%d best=%.4f th=%.2f v=%.1f td=%.2f tau=%.2f (%.1fs)\n', ...
            i,nAll,best,mod(bestX(1)*180/pi,360),bestX(2),bestX(3),bestX(4),toc(tC));
        plog(logfid, msg);
        % 中间最优落盘
        save(fullfile(outDir,'q2_best_so_far.mat'),'best','bestX','i','durs','X');
    end
end
[ds,ord]=sort(durs,'descend'); Xs=X(ord,:);
plog(logfid, sprintf('粗搜完成 %.1fs top=%.4f\n', toc(tC), ds(1)));
for k=1:min(10,nAll)
    plog(logfid, sprintf('  #%d %.4f th=%.2f v=%.1f td=%.3f tau=%.3f\n', ...
        k,ds(k),mod(Xs(k,1)*180/pi,360),Xs(k,2),Xs(k,3),Xs(k,4)));
end
save(fullfile(outDir,'q2_coarse.mat'),'Xs','ds','bestX','best','dur_q1r');

% fmincon
nTop=15; nRef=4e4;
plog(logfid, sprintf('---- fmincon Top-%d ----\n', nTop));
opts=optimoptions('fmincon','Display','off','Algorithm','sqp', ...
    'MaxFunctionEvaluations',400,'MaxIterations',100, ...
    'OptimalityTolerance',1e-8,'StepTolerance',1e-12,'UseParallel',false);
obj=@(x)-objective_q2(x,p,nRef);
cX=Xs(1:min(nTop,nAll),:); cXf=cX; cD=zeros(size(cX,1),1);
tR=tic;
for k=1:size(cX,1)
    try
        [xf,fv]=fmincon(obj,cX(k,:),[],[],[],[],lb,ub,[],opts);
        cXf(k,:)=xf; cD(k)=-fv;
    catch ME
        plog(logfid, sprintf('  fail#%d %s\n',k,ME.message));
        cD(k)=objective_q2(cX(k,:),p,nRef); cXf(k,:)=cX(k,:);
    end
    plog(logfid, sprintf('  %d/%d -> %.4f (from %.4f)\n',k,size(cX,1),cD(k),ds(k)));
end
plog(logfid, sprintf('精修完成 %.1fs\n', toc(tR)));
[bm,ib]=max(cD); xbest=cXf(ib,:);
plog(logfid, sprintf('精修最优 %.6f\n', bm));

% patternsearch
try
    ops=optimoptions('patternsearch','Display','off','MaxFunctionEvaluations',600, ...
        'UseCompletePoll',true,'UseParallel',false,'MeshTolerance',1e-9);
    [xps,~]=patternsearch(@(x)-objective_q2(x,p,5e4),xbest,[],[],[],[],lb,ub,[],ops);
    dps=objective_q2(xps,p,6e4);
    plog(logfid, sprintf('PS %.6f\n', dps));
    if dps>bm+1e-4, xbest=xps; bm=dps; plog(logfid,'采用PS\n'); end
catch ME
    plog(logfid, sprintf('PS skip %s\n', ME.message));
end

% 高精度
nHi=3e5;
[~,info]=objective_q2(xbest,p,nHi);
[dur_ref,tin,tout]=shield_duration(info.P_det,info.t_det,p,nHi,true);
info.dur=dur_ref; info.t_in=tin; info.t_out=tout;

plog(logfid, sprintf('\n==== 最优 ====\n时长=%.6f [% .6f, %.6f]\n',dur_ref,tin,tout));
plog(logfid, sprintf('theta=%.6f (%.4f deg) v=%.6f\n',info.theta,info.heading_deg,info.v));
plog(logfid, sprintf('td=%.6f tau=%.6f tdet=%.6f\n',info.t_drop,info.tau,info.t_det));
plog(logfid, sprintf('Pdrop=[%.6f %.6f %.6f]\n',info.P_drop));
plog(logfid, sprintf('Pdet=[%.6f %.6f %.6f]\n',info.P_det));
plog(logfid, sprintf('Q1 %.4f -> %.4f\n',dur_q1r,dur_ref));

% 正式结果
fid=fopen(fullfile(outDir,'q2_result.txt'),'w','n','UTF-8');
fprintf(fid,'==== 2025国赛A题 问题2 结果 ====\n');
fprintf(fid,'判据: 云团球心到导弹-真目标中心视线段距离 <= 10 m\n');
fprintf(fid,'有效遮蔽时长 = %.6f s\n',dur_ref);
fprintf(fid,'有效区间(首段) = [%.6f, %.6f] s\n',tin,tout);
fprintf(fid,'\n--- 决策变量 ---\n');
fprintf(fid,'航向角 theta = %.10f rad = %.6f deg\n',info.theta,info.heading_deg);
fprintf(fid,'飞行速度 v   = %.10f m/s\n',info.v);
fprintf(fid,'投放时刻 t_drop = %.10f s\n',info.t_drop);
fprintf(fid,'引信延时 tau    = %.10f s\n',info.tau);
fprintf(fid,'\n--- 导出量 ---\n');
fprintf(fid,'速度向量 v_fy = [%.10f, %.10f, %.10f] m/s\n',info.v_fy);
fprintf(fid,'投放点 P_drop = [%.10f, %.10f, %.10f] m\n',info.P_drop);
fprintf(fid,'起爆时刻 t_det = %.10f s\n',info.t_det);
fprintf(fid,'起爆点 P_det  = [%.10f, %.10f, %.10f] m\n',info.P_det);
fprintf(fid,'\n--- 对照 ---\n');
fprintf(fid,'问题1固定策略时长 = %.6f s\n',dur_q1r);
fprintf(fid,'提升 Delta = %.6f s\n',dur_ref-dur_q1r);
fprintf(fid,'粗搜样本数 = %d, 精修候选 = %d\n',nAll,size(cX,1));
fprintf(fid,'总运行时间 = %.2f s\n',toc(tAll));
fclose(fid);

save(fullfile(outDir,'q2_result.mat'),'xbest','info','dur_ref','tin','tout','dur_q1r','cXf','cD','Xs','ds','p');
fid=fopen(fullfile(outDir,'q2_result.csv'),'w','n','UTF-8');
fprintf(fid,'theta_rad,heading_deg,v_mps,t_drop_s,tau_s,t_det_s,Pdrop_x,Pdrop_y,Pdrop_z,Pdet_x,Pdet_y,Pdet_z,duration_s,t_in,t_out\n');
fprintf(fid,'%.10f,%.6f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n', ...
    info.theta,info.heading_deg,info.v,info.t_drop,info.tau,info.t_det, ...
    info.P_drop(1),info.P_drop(2),info.P_drop(3),info.P_det(1),info.P_det(2),info.P_det(3),dur_ref,tin,tout);
fclose(fid);

plog(logfid, sprintf('结果已写入, 总耗时 %.1fs\nDONE\n', toc(tAll)));
fclose(logfid);
end

function plog(fid, msg)
fprintf('%s', msg);
if fid>0
    fprintf(fid, '%s', msg);
    try, fflush(fid); catch, end
end
drawnow('update');
end
