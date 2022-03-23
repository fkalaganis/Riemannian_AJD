%% Visualize Sources for EEG generation
clear all, close all
load emptyEEG
Fs=1000;
% Source Locations
chanlocX=EEG.lf.GridLoc(:,1)';
chanlocY=EEG.lf.GridLoc(:,2)';
chanlocZ=EEG.lf.GridLoc(:,3)';
% Source IDs
for i_source=1:2004
    sourceIDs{i_source}=i_source;
end

K_neigh=30;
[dipoleC1,dipoleC1_dist] = knnsearch(EEG.lf.GridLoc,EEG.lf.GridLoc(255,:),'K',K_neigh)
weightC1=1-dipoleC1_dist*40;
[dipoleC2,dipoleC2_dist] = knnsearch(EEG.lf.GridLoc,EEG.lf.GridLoc(257,:),'K',K_neigh)
weightC2=1-dipoleC2_dist*40;

c=colormap(autumn(K_neigh));
c=c(end:-1:1,:);

figure,scatter3(chanlocX, chanlocY, chanlocZ,50,'red','filled'), hold on
scatter3(chanlocX(dipoleC1), chanlocY(dipoleC1), chanlocZ(dipoleC1),50,c,'filled')
%dipoleC2=[1767 1741 1571 1677 1939 1786];
scatter3(chanlocX(dipoleC2), chanlocY(dipoleC2), chanlocZ(dipoleC2),50,c,'filled')

trial_no=25;
%% Generate Synthetic Signals
% EEG signals of the first class
dipoleLoc=dipoleC1;
[b,a]=butter(3, [8 12]/(Fs/2));
for i_trial=1:trial_no
    [1 i_trial]
    whitenoise=randn(1,Fs*2);
    alphanoise=filtfilt(b,a,whitenoise);
    dipoleSignals=repmat(alphanoise,length(dipoleC1),1);
    dipoleSignals=weightC1'.*dipoleSignals;
    synthEEG=synthesizeEEG(dipoleLoc, dipoleSignals, 'pink', [], 200);
    EEGdata{i_trial}=synthEEG.data;
end
% EEG signals of the second class
dipoleLoc=dipoleC2;
[b,a]=butter(3, [8 12]/(Fs/2));
for i_trial=1:trial_no
    [2 i_trial]
    whitenoise=randn(1,Fs*2);
    alphanoise=filtfilt(b,a,whitenoise);
    dipoleSignals=repmat(alphanoise,length(dipoleC2),1);
    dipoleSignals=weightC2'.*dipoleSignals;
    synthEEG=synthesizeEEG(dipoleLoc, dipoleSignals, 'pink', [], 200);
    EEGdata{end+1}=synthEEG.data;
end
% Labels for each Trial
labels=[zeros(1, trial_no) ones(1, trial_no)]'+1;

%% Calculate Covariances
for i_trial=1:size(EEGdata,2)
    filtered_covariances{i_trial}=cov(EEGdata{i_trial}');% I keep only the first ten seconds. You can try more options
end

%% Jointly Diagonalize Covariance matrices
[V,qDs]= joint_diag([filtered_covariances{:}],1.0000e-08);%Set 1
for i=1:size(filtered_covariances,2)
    reconstructed_covariances{i}=diag(diag(V'*filtered_covariances{i}*V));%/trace(myV*diag(diag(myV'*filtered_covariances{i}*myV))))*myV';
    reconstructed_covariances_array(:,:,i)=reconstructed_covariances{i};
    lambdas(i,:)=diag(V'*filtered_covariances{i}*V);%/sum(diag(myV'*filtered_covariances{i}*myV));
end

%% PLOT
% Initial covariance space
figure,subplot(1,2,1),mydist=zeros(size(filtered_covariances,2),size(filtered_covariances,2));
for i_trial=1:size(filtered_covariances,2)
    for j_trial=1:size(filtered_covariances,2)-1
        mydist(i_trial,j_trial)=(distance_riemann(squeeze(filtered_covariances{i_trial}),squeeze(filtered_covariances{j_trial})));
        mydist(j_trial,i_trial)=mydist(i_trial,j_trial);
    end
end
for i_trial=1:size(filtered_covariances,2)
    mydist(i_trial,i_trial)=0;
end
Y = cmdscale(mydist);
Dtriu = mydist(find(tril(ones(length(mydist)),-1)))';
maxrelerr(1) = max(abs(Dtriu-pdist(Y(:,1:48))))./max(Dtriu)
colours=[1 0 0; 0 0 1; 0 1 0];% red->sad, blue->neutral, green->happy
for i_trial=1:size(labels,1)
    if labels(i_trial)==0
        continue;
    end
    scatter(Y(i_trial,1),Y(i_trial,2),[],colours(labels(i_trial)+1,:),'filled'), hold on
    
    text(Y(i_trial,1),Y(i_trial,2),num2str(i_trial))
end, title('Init Cov Space'), axis equal

% JD covariance Space
subplot(1,2,2),mydist=zeros(size(reconstructed_covariances,2),size(reconstructed_covariances,2));
for i_trial=1:size(reconstructed_covariances,2)
    for j_trial=1:size(reconstructed_covariances,2)-1
        mydist(i_trial,j_trial)=(distance_riemann(squeeze(reconstructed_covariances{i_trial}),squeeze(reconstructed_covariances{j_trial})));
        mydist(j_trial,i_trial)=mydist(i_trial,j_trial);
    end
end
for i_trial=1:size(reconstructed_covariances,2)
    mydist(i_trial,i_trial)=0;
end
Y = cmdscale(mydist);
Dtriu = mydist(find(tril(ones(length(mydist)),-1)))';
maxrelerr(1) = max(abs(Dtriu-pdist(Y(:,1:48))))./max(Dtriu)
colours=[1 0 0; 0 0 1; 0 1 0];% red->sad, blue->neutral, green->happy
for i_trial=1:size(labels,1)
    if labels(i_trial)==0
        continue;
    end
    scatter(Y(i_trial,1),Y(i_trial,2),[],colours(labels(i_trial)+1,:),'filled'), hold on
    
    text(Y(i_trial,1),Y(i_trial,2),num2str(i_trial))
end, title('AJD Cov Space'), axis equal


for i=1:trial_no*2
    mycov(i,:,:)=filtered_covariances{i};
    myEEG(i,:)=mean(EEGdata{i}.^2,2);
end
C1=squeeze(mean(mycov(1:trial_no,:,:)));
C2=squeeze(mean(mycov(trial_no+1:trial_no*2,:,:)));

[W1,L1] = eig(C1);
[evals,sidx] = sort(diag(L1),'descend');
W1 = W1(:,sidx);
map1=W1(:,1)'*C1;

[W2,L2] = eig(C2);
[evals,sidx] = sort(diag(L2),'descend');
W2 = W2(:,sidx);
map2=W2(:,1)'*C2;

figure, subplot(1,2,1), topoplotIndie(map1,EEG.chanlocs,'numcontour',0,'electrodes','off','plotrad',.6);
subplot(1,2,2), topoplotIndie(map2,EEG.chanlocs,'numcontour',0,'electrodes','off','plotrad',.6);

