
#/home/liying_lab/chenxinfeng/ml-project/LILAB-py/lilab/photometry
#su chenxinfeng  
#conda activate mmdet  #or OpenLabCluster
##numpy, scipy, h5py, pickle, detectTTL


##fiber photometry path
#data_dir=/mnt/liying.cibr.ac.cn_220T_Data_Temp_ZZC_bak/SocialPhot_BatchesDemo
#fd=Batch5/phot_raw  
#/mnt/liying.cibr.ac.cn_220T_Data_Temp_ZZC_bak/202602_Suc-Quine/WT
#test

#data_dir=/mnt/liying.cibr.ac.cn_220T_Data_Temp_ZZC_bak/Batch6-WTfemale-wm202508
#fd=phot    
#path=$data_dir/$fd

path=/mnt/liying.cibr.ac.cn_220T_Data_Temp_ZZC_bak/202602_Suc-Quine/202603-sucQuin-S3
###step 1, extract raw data,
#echo "$path" | xargs python -m lilab.photometry.s1_tdms_2_photpkl  #solve .tdms file
echo "$path" | xargs python -m lilab.photometry.s1_fipgui_2_photpkl   ##solve .mat file
###step 2, signal processing, smooth, motion correction, dF/F or z-score
echo "$path" | xargs python -m lilab.photometry.s2_photpkl_2_dFF      

###step 3, choose signal Channel
ibrain=0 # 0,one side; 0|1，two sides, ##brain region, #left-0 or right-1 in phot.jpg
ichan=1 #channel of dopamine, various large channel #signal Channel, 0=405(reference)，1=470 (Green) ，2=565 (Red)    ###*photpick.jpg shows ichan signal, add data_sig in *.dFFphotpkl
echo "$path" | xargs -I {} python -m lilab.photometry.s4_dFF_repick {} $ibrain $ichan
