#!/bin/bash
UNFOLD_DIR=/home/fynu/mkomm/stpol/unfold/tunfold17.3
LD_LIBRARY_PATH=$UNFOLD_DIR:$LD_LIBRARY_PATH

SYS="comphep_nominal"
CHANNEL="mu"
./unfoldPro \
--histFile="/nfs/user/mkomm/scanned_hists_jul31/0.60000/"$CHANNEL"/merged/cos_theta_lj.root" \
--responseFile="/nfs/user/mkomm/scanned_hists_jul31/0.60000/"$CHANNEL"/tmatrix_nocharge__gen_ele.root" \
--responseFile="/nfs/user/mkomm/scanned_hists_jul31/0.60000/"$CHANNEL"/tmatrix_nocharge__gen_mu.root" \
--responseFile="/nfs/user/mkomm/scanned_hists_jul31/0.60000/"$CHANNEL"/tmatrix_nocharge__gen_tau.root" \
--fitResultPrefix="/nfs/user/mkomm/scanned_hists_jul31/0.60000/fitresult" \
--fitResult=$CHANNEL".txt" \
--fitCovariance=$CHANNEL"_cov.root" \
--sys=$SYS \
--output=$CHANNEL"_"$SYS".root" \
-v \
--no-stat \
--no-mcstat \
--no-fiterror 
