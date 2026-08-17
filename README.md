1. Rats' social behavior classification based on 3D skeletons
2. Dopamine photometry signal analysis
3. Social syllable post-process
4. Align social syllable and dopamine signal.


Data path: Batch3, Batch5
Script path:  socialClassify-OnsetRefine_photAlign_pipeline
Run in turn:
       1_socialClassify-photPreprocess:  social behaivor classification  01_1_socialClassifyEachFolder-speedNormBymultiFolders.sh;
                                         wire fiber photometry preprocess  02_1_photom-wire-process.sh
                                         wireless photometry preprocess 02_2_photom-wireless_process.sh
       2_socialOnsetRefine_photAlign:  001_socialBehaviorOnsetRefine_alignPhotometry.sh
