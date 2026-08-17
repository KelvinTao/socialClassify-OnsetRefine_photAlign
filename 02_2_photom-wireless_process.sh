
#su chenxinfeng
#source activate mmdet #or OpenLabCluster
##numpy, scipy, h5py, pickle, detectTTL


###wireless photometry files in one folder
##solve .h5 file
data_dir=/mnt/liying.cibr.ac.cn_220T_Data_Temp_ZZC_bak/SocialPhot_BatchesDemo
fd=Batch3/phot_raw
path=$data_dir/$fd
##1 or multi-recordings in 1 wireless h5 file, >2 min
## signal processing, jump detection, smoothen.  dF/F
echo "$path" | xargs python -m lilab.photometry.wireless_dFF_multiOr1record_tao20250313  
