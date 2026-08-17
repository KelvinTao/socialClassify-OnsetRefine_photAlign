
####script path /home/liying_lab/chenxinfeng/ml-project/LILAB-py/lilab

#------ Classify behaivor label 0.8s clip with 0.1s step ------
#------ labeling on the first frame of the 0.8s clip ------
#su chenxinfeng
conda activate OpenLabCluster
####data root dir
data_dir=/mnt/liying.cibr.ac.cn_220T_Data_Temp_ZZC_bak/SocialPhot_BatchesDemo 
folders=(Batch3 Batch5) #precessed folders

#######calculate raw features in each folder
folders_str=""
for fd in ${folders[@]}
    do
        echo $fd
        fd2=$fd/3D-mp4
        folders_str="$folders_str,$fd2"
        ## bodylength.pkl,rawfeat.pkl
        python -m lilab.lstm_bhv_bodylennorm_classify.s0_prepare_body_length $data_dir/$fd2
        python -m lilab.lstm_bhv_bodylennorm_classify.s01_matcalibpkl2rawfeatpkl $data_dir/$fd2
    done

##Back speed normalization and classification
gpu_i=0
for fd in ${folders[@]}
    do
        res_dir=$data_dir/$fd/3D-mp4
        ###!!! Back speed normalization By one or multi-folder speed !!!
        python -m lilab.lstm_bhv_bodylennorm_classify.s02_rawfeatpkl_to_norm_ByMultiFolders_tao  $data_dir  $folders_str  $res_dir  #normalization by one or more folders
        ##python -m lilab.lstm_bhv_bodylennorm_classify.s02_rawfeatpkl_to_norm  $res_dir  #self folder normalization
        #output: rawfeat_norm.pkl

        ### behavior classification using speed normalized features
        ### --use-normed using normalized speed, --stride 3: 10 Hz prediction, for Dopamine signal analysis; ----stride 24: 0.8s for behavior comparison
        CUDA_VISIBLE_DEVICES=$((gpu_i%4))  python -m lilab.lstm_bhv_bodylennorm_classify.s1_matcalibpkl2clippredpkl_semiseq2seq $res_dir --use-normed   --stride 3  &
        ###output: $res_dir/out_semiseq2seq_norm/lstm_offline.clippredpkl
        
        ###  !!! Unnecessary for photometry analysis!!!
        ### keep consistent clips (mirrored for 2 rats)，  representative; Suitable for behavior comparison
        ###python -m lilab.OpenLabCluster_train.a1_mirror_mutual_filt_clippredpkl $res_dir/out_semiseq2seq_norm/lstm_offline.clippredpkl --already-mirrored &
        ##output: $res_dir/out_semiseq2seq_norm/representitive_k36_filt_perc*/Representive_K36.clippredpkl
       
        gpu_i=$((gpu_i+1))
    done

# plot video examples of each class
su chenxf
conda activate mmdet
data_dir=/mnt/liying.cibr.ac.cn_220T_Data_Temp_ZZC_bak/Batches-demo
res_dir=$data_dir/Batch5/3D-mp4
python -m lilab.OpenLabCluster_train.a6b_clippredpkl_2_cluster400p  $res_dir/out_semiseq2seq_norm/representitive_k36_filt_perc*/Representive_K36.clippredpkl $res_dir
##output: $res_dir/out_semiseq2seq_norm/representitive_k36_filt_perc*/400p_clusters

