#/home/liying_lab/chenxinfeng/ml-project/LILAB-py/lilab/photometry

####### 2. generate raw features #######
#su chenxinfeng  
#conda activate mmdet  #or OpenLabCluster

##
data_dir=/mnt/liying.cibr.ac.cn_220T_Data_Temp_ZZC_bak/SunLe ##data root dir
folders=(analyzing) #precessed folders

for fd in ${folders[@]}
    do
        echo $fd
        PROEJECT_DIR=$data_dir/$fd    ##/3D-mp4
        ## 1. generate 24 raw features (overlap no sign), 30 Hz
        python -m lilab.photometry.002-1_SimplifyFeats24fromPoints-IDswitch_mp $PROEJECT_DIR
        ##output: $PROEJECT_DIR/3D-mp4/feats24-0.8s_nosign/*First.pkl
        ## 2. feature normalization and save in a long matrix (length 30Hz ,24)
        python -m lilab.photometry.002-2_behavior-feats-wholeLength  $PROEJECT_DIR
        ##output: $PROEJECT_DIR/3D-mp4/feats24-0.8s_nosign/*.f24pkl
        ####### 3. match features--behavior lables #######
        python -m lilab.photometry.003_featureProcess-mid_sl_taoStep24  $PROEJECT_DIR  #step 24 predicted labels
        ##output: $PROEJECT_DIR/3D-mp4/out_semiseq2seq_norm/bhv36normV-f24nosign-bhvMerge7,  bhvDicts.pkl, k36bhvMerge7-eachRat-feat24.pkl

        ### 4. refine behavior labels by the paired feature
        python -m lilab.photometry.004_bhvMerge7-bhvOnsetRefine_sl_tao $PROEJECT_DIR
        ##output: $PROEJECT_DIR/3D-mp4/out_semiseq2seq_norm/bhv36normV-f24nosign-bhvMerge7/bhvRefined-events.pkl
        ##dataframe, video, bhv, startframe 10 Hz, duration...

        ##align photometry and social behavior events
        #PROEJECT_DIR=$data_dir/$fd
        #python -m lilab.photometry.010_alignBhv-phot $PROEJECT_DIR
        ##output: $PROEJECT_DIR/3D-mp4/out_semiseq2seq_norm/bhv36normV-f24nosign-bhvMerge7/photometry_bhvRefined-events.pkl
    done



