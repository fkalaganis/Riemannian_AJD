% clear; close all
% Fs=1000;
% [b,a]=butter(3, [8 12]/(Fs/2));
% whitenoise=randn(1,Fs*3);
% alphanoise=filtfilt(b,a,whitenoise);
% plot(whitenoise), hold on
% plot(alphanoise*10), hold on
% t=[1:length(alphanoise)]-round(length(alphanoise)/2);
% s=1./(1+exp(-t/600));
% plot(alphanoise.*s(end:-1:1)*10)

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
% figure, subplot(1,2,1),scatter3(chanlocX, chanlocY, chanlocZ);
% text(chanlocX, chanlocY, chanlocZ, sourceIDs)
% subplot(1,2,2),scatter(chanlocX, chanlocY);
% text(chanlocX, chanlocY, sourceIDs)
%% Generate Synthetic Signals
% EEG signals of the first class
dipoleLoc=1838;
[b,a]=butter(3, [8 12]/(Fs/2));

for i_trial=1:50
    whitenoise=randn(1,Fs*3);
    alphanoise=filtfilt(b,a,whitenoise);
    t=[1:length(alphanoise)]-round(length(alphanoise)/2);
    s=1./(1+exp(-t/600));
    dipoleSignal=alphanoise.*s(end:-1:1)*10;
    %     subplot(2,1,1),plot(dipoleSignal)
    %     subplot(2,1,2),plot(s)
    synthEEG=synthesizeEEG(dipoleLoc, dipoleSignal, 'pink', [], 200);
    EEGdata{i_trial}=synthEEG.data;
end
% EEG signals of the second class
dipoleLoc=1838;
[b,a]=butter(3, [8 12]/(Fs/2));
for i_trial=1:50
    whitenoise=randn(1,Fs*3);
    alphanoise=filtfilt(b,a,whitenoise);
    t=[1:length(alphanoise)]-round(length(alphanoise)/2);
    s=1./(1+exp(-t/600));
    dipoleSignal=alphanoise.*s(1:end)*10;
    synthEEG=synthesizeEEG(dipoleLoc, dipoleSignal, 'pink', [], 200);
    EEGdata{end+1}=synthEEG.data;
end
% Labels for each Trial
labels=[zeros(1, 50) ones(1, 50)]'+1;

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
end, title('JD Cov Space'), axis equal

%% Segment Epochs in TV-COVs
STEP_SIZE=round(Fs/10);
WINDOW_SIZE=round(Fs);
TV_covs={};
TV_covs_all={};
TV_covs_array=[];
for i=1:size(filtered_covariances,2)
    mycount=1;
    for j=round((WINDOW_SIZE+1)/2):STEP_SIZE:size(EEGdata{1},2)-round((WINDOW_SIZE+1)/2)
        myCOV=cov((EEGdata{i}(:,j-round((WINDOW_SIZE+1)/2)+1: j+round((WINDOW_SIZE+1)/2)))');
        TV_covs{i,mycount}=diag(diag(V'*myCOV*V));
        TV_covs_array(:,:,end+1)=diag(diag(V'*myCOV*V));
        TV_covs_all{end+1}=diag(diag(V'*myCOV*V));
        mycount=mycount+1;
    end
end
TV_covs_array(:,:,1)=[];
%% Calculate the lexicon for VQ

clust=[];for i=2:10
    i
    my_spd_matrices=spd_initialize(TV_covs_array);
    [clust(:,i), C, cost] = spd_kmeans(my_spd_matrices, i);
end

for i_clust=2:10
    kept_clust=clust(:,i_clust);
    %     prot=[];for i=unique(kept_clust)' % Calculate the prototypes
    %         prot(end+1,:)=diag(karcher(TV_covs_all{kept_clust==i}));
    %     end
%     figure(i_clust+100),subplot(2,1,1),hist(kept_clust(1:400),[0.5:1:i_clust])
%     subplot(2,1,2),hist(kept_clust(401:800),[0.5:1:i_clust])
end

for i_clust=4%:10;
    kept_clust=clust(:,i_clust);
    total_bin_counts=[];
    total_symbolic_series=[];
    for i_segment=0:size(TV_covs,2):length(TV_covs_all)-size(TV_covs,2)
        [bin_counts, ~]=hist(kept_clust(i_segment+1:i_segment+16),[0.5:1:i_clust]);
        total_bin_counts(end+1,:)=bin_counts;
        total_symbolic_series(end+1,:)=(kept_clust(i_segment+1:i_segment+20));
    end
    %% Re-asign color for better visualization
%     total_symbolic_series=total_symbolic_series*10;
%     total_symbolic_series(total_symbolic_series==10)=4;
%     total_symbolic_series(total_symbolic_series==40)=3;
%     total_symbolic_series(total_symbolic_series==20)=2;
%     total_symbolic_series(total_symbolic_series==30)=1;
    figure(i_clust+200), subplot(2,1,1), imagesc(total_symbolic_series)
    xticks([1:1:20])
    xticklabels({0.5:0.1:3})
    xlabel('Time (seconds)')
    subplot(2,1,2), imagesc(total_bin_counts)
end

for i_cov=1:2000
    mynewcov{i_cov}=squeeze(TV_covs_array(:,:,i_cov));
end

