function [synth_EEG]=synthesizeEEG_noSources(dipoleLoc, dipoleSignals, noSources, method, covStructure, signalStrength)
% --- DESCRIPTION---
%Generates synthetic EEG data using a forward model. The data are generated
%with a 1000Hz sapling frequency.
% ---INPUT---
%   *dipoleLoc: A vector containg the indices of dipoles (range 1-2004)
%   *dipoleSignals: A mtrix contraining the signals for the dipoles (should be
% in the form of dipole_number x time_samples)
%   *method: background noise, 'pink' or 'white' (default)
%   *covStructure: Covariance structure to rule the sources (default is I)
%   *signalStrength: Strength of the given signals
%--- OUTPUT ---
%   *synth_EEG: A struct that contains all the essential informations for
%  the synthetic EEG. The recording is under the name EEG.data.
%
% Created by Fotis P. Kalaganis, 03 Dec 2021




%% preliminaries
EXP_DECAY=200; % exponential decay parameter for pink noise
load emptyEEG % mat file containing EEG, leadfield and channel locations

%% Sanity Checks and Initializations
if length(dipoleLoc)~=size(dipoleSignals,1)
    synth_EEG=[];
    disp('NUmber of signals and sources should be equal')
    return
end

if isempty(method)
    noSources=size(EEG.lf.Gain,3);
end
tempVar=randperm(size(EEG.lf.Gain,3));
mySources=tempVar(1:noSources-1)';
mySources=cat(1, dipoleLoc, mySources);

if isempty(method)
    method='white'
end

if isempty(covStructure)
    covStructure=eye(size(EEG.lf.Gain,3));
end

if isempty(signalStrength)
    signalStrength=200;
end

if ~isempty(dipoleLoc)
    EEG.pnts=size(dipoleSignals,2);
    EEG.times=[1:EEG.pnts]/EEG.srate;
else
    EEG.pnts=60000;
    EEG.times=[1:EEG.pnts]/EEG.srate;
end


%% create the dipole time series
L=chol(covStructure);
if strcmp(method,'white') %background white noise
    dipdat = L*randn(size(EEG.lf.Gain,3),EEG.pnts)*3;
else
    for di=1:size(EEG.lf.Gain,3)
        as = rand(1,EEG.pnts) .* exp(-(0:EEG.pnts-1)/EXP_DECAY);
        data = real(ifft(as .* exp(1i*2*pi*rand(size(as)))));
        dipdat(di,:) = (data-mean(data))/std(data);
    end
    dipdat=L*dipdat;
end

for di=1:length(dipoleLoc) %put in dipole time series
    dipdat(dipoleLoc(di),:) = dipoleSignals(di,:)*signalStrength;
end
zeroSources=setdiff(mySources,[1:size(EEG.lf.Gain,3)]);
dipdat(zeroSources,:)=0;

EEG.data = squeeze(EEG.lf.Gain(:,1,:))*dipdat;
synth_EEG=EEG;


end
