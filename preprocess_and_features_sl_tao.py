
#conda activate DLC-GPU2.2-taoxm
#conda activate bcnet_bak_20211105
#conda activate base
#python
import numba
import seaborn as sns
from inspect import cleandoc as dedent
import scipy.io as sio
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from numpy import random
from scipy import ndimage
from sklearn import preprocessing,decomposition
import pywt
from os.path import join

##z-score and PCA
def get_zscore(data):
    #zscore
    zscore=preprocessing.StandardScaler()
    data_zs=zscore.fit_transform(data)
    return(data_zs)

def get_zs_pca(data,n_components):
    #zscore
    zscore=preprocessing.StandardScaler()
    data_zs=zscore.fit_transform(data)
    ##pca
    pca=decomposition.PCA(n_components=n_components)
    pca_model=pca.fit(data_zs)
    data_zs_pca=pca_model.transform(data_zs)
    return(data_zs_pca)


##pca
def get_pca(data,n_components):
    pca=decomposition.PCA(n_components=n_components)
    pca_model=pca.fit(data)
    data_pca=pca_model.transform(data)
    return(data_pca)


#0-13:
#'nose','earL','earR','neck','body','tail','shoulderL','palmL','shoulderR','palmR','kneeL','footL','kneeR','footR'
#'0-nose','1-earL','2-earR','3-neck','4-body','5-tail','6-shoulderL','8-shoulderR','10-kneeL','12-kneeR'
##'7-palmL','9-palmR','11-footL','13-footR'

##interaction feature calculation
#distance between points
def distance(p1,p2):
    #calculate distance between p1 and p2
    dist=np.dot(p1-p2,p1-p2)**0.5
    return(dist)

def distance_vect(vect):
    #calculate length of vect from zero point
    dist=np.dot(vect,vect)**0.5
    return(dist)

####preprocess####
def getHeadCenter(ratB):
    return(np.mean(ratB[:,[0,1,2]],axis=1))

#angle of 3-10 edge pairs
def getTrunkCenter(ratB):
    cents=np.mean(ratB[:,[3,4,5,6,8,10,12]],axis=1)
    return(cents)

def get_3partsCenter(ratB):
    ##head center, TrunkCenter_Neck, TrunkCenter_Tail
    points3=np.stack((np.mean(ratB[:,[0,1,2]],axis=1),
            np.mean(ratB[:,[3,6,8]],axis=1),
            np.mean(ratB[:,[5,10,12]],axis=1)),
            axis=1)
    return(points3)


def median_filter_5frames(ratB):
    ratB=ndimage.median_filter(ratB, size=(5,1,1)) ##filter by frames
    return(ratB)

    
## 3D translate by trunkCenter
## horizontal rotation(x-y) rotate by trunkC2tailBase
## rescale by nose-tailbase length mean

def angle_vectors2(vector1,vector2):
    #calculate angle between vectors
    numerator=np.dot(vector1, vector2)
    denominat=np.dot(vector1,vector1)**0.5*np.dot(vector2,vector2)**0.5
    if denominat==0:denominat=0.00001
    cosTheta=round(numerator/denominat,10)
    cosTheta=1 if cosTheta>1 else cosTheta
    cosTheta=-1 if cosTheta<-1 else cosTheta
    return(math.acos(cosTheta)*180/math.pi)

##0-180
def angle_vectors(vector1,vector2):
    #calculate angle between vectors
    numerator=np.dot(vector1, vector2)
    denominat=np.dot(vector1,vector1)**0.5*np.dot(vector2,vector2)**0.5
    if denominat==0:denominat=0.00001
    cosTheta=round(numerator/denominat,10)
    cosTheta=-1 if cosTheta<-1 else cosTheta
    cosTheta=1 if cosTheta>1 else cosTheta
    return(math.acos(cosTheta)*180/math.pi)


def get_point_velocity(cents,fps): #one 3D point velicity
    #cents:(frame,dim=3)
    #cents_v:(frame,dim=3)
    time_gap=1/fps; #fps=15
    cents_v1=np.diff(cents,axis=0)/time_gap
    cents_v=np.concatenate([np.expand_dims(cents_v1[0,:],axis=0),cents_v1],axis=0)
    return(cents_v)


def get_speed_from_velocity(cents_v):
    speed=np.array([np.dot(a,a)**0.5 for a in cents_v])
    return(speed)


def get_speed_from_velocity_multipoints(cents_v):
    f,p=cents_v.shape[:2]
    speed=np.zeros((f,p))
    for fi in range(f):
        for pi in range(p):
            speed[fi,pi]=np.dot(cents_v[fi,pi],cents_v[fi,pi])**0.5
    return(speed)


import pandas as pd
def moving_mean_sd_frames(cents_v,frame_window):
    ##frame_window=3
    dim=cents_v.shape[1]
    mm=np.array([pd.Series(cents_v[:,j]).rolling(window=frame_window,center=True,axis=0, min_periods=1).mean() for j in range(dim)]).T
    ms=np.array([pd.Series(cents_v[:,j]).rolling(window=frame_window,center=True,axis=0, min_periods=1).var(ddof=0) for j in range(dim)]).T
    ##pandas std: default ddof=1, ddof=0 in numpy
    return([mm,ms])



####interaction features####
##0-13:nose,earL,earR,neck,body,tail,
#shoulderL,palmL,shoulderR,palmR,kneeL,footL,kneeR,footR
##interaction feature calculation

def angle_lineBypoints(line1_p1,line1_p2,line2_p1,line2_p2):
    #calculate angle between vectors calculated by points:p2->p1
    vector1=line1_p1-line1_p2#3d
    vector2=line2_p1-line2_p2
    return(angle_vectors(vector1,vector2))



#get distance of each two points between rats
def get_distances_BetweenPoints(ratB,ratW):
    frame_num,point_num=ratB.shape[0:2]
    ##
    points_dists=np.zeros((frame_num,point_num,point_num))
    for fi in range(frame_num):
        for pbi in range(point_num):
            for pwi in range(point_num):
                points_dists[fi,pbi,pwi]=distance(ratB[fi,pbi],ratW[fi,pwi])
    return(points_dists)
    


def get_distances_BetweenPoints2(ratB,ratW):
    frame_num,point_num=ratB.shape[0:2]
    point_num2=ratW.shape[1]
    ##
    points_dists=np.zeros((frame_num,point_num,point_num2))
    for fi in range(frame_num):
        for pbi in range(point_num):
            for pwi in range(point_num2):
                points_dists[fi,pbi,pwi]=distance(ratB[fi,pbi],ratW[fi,pwi])
    return(points_dists)

##calculate distances velicity between points (frame_num,14,14,4)

def nose2tail_length_median(ego3D):
    bsize=np.zeros((ego3D.shape[0],1))#(frame,part_num,3)
    for i in range(ego3D.shape[0]):
        bsize[i]=distance(ego3D[i,0],ego3D[i,5])##nose2tail distance
    return(np.median(bsize))


def get_bodyLen_2rats(ratB,ratW):
    len_B=nose2tail_length_median(ratB)
    len_W=nose2tail_length_median(ratW)
    len_body=np.mean([len_B,len_W])
    return(len_body)


def get_near_indexes(ratB,ratW,min_distance_body_times):
    points_dists=get_distances_BetweenPoints(ratB,ratW)
    f_num=points_dists.shape[0]
    len_B=nose2tail_length_median(ratB)
    len_W=nose2tail_length_median(ratW)
    len_body=np.mean([len_B,len_W])
    close_inds=np.zeros(f_num)
    for fi in range(f_num):
        if np.min(points_dists[fi])<=min_distance_body_times*len_body: ##min_distance 
            close_inds[fi]=1
    return(close_inds)

def get_far_indexes(ratB,ratW,min_distance_body_times):
    points_dists=get_distances_BetweenPoints(ratB,ratW)
    f_num=points_dists.shape[0]
    len_B=nose2tail_length_median(ratB)
    len_W=nose2tail_length_median(ratW)
    len_body=np.mean([len_B,len_W])
    close_inds=np.zeros(f_num)
    for fi in range(f_num):
        if np.min(points_dists[fi])>min_distance_body_times*len_body: ##>min_distance 
            close_inds[fi]=1
    return(close_inds)


def duration(st):
    ## state labels
    st_num=np.unique(st)
    st_num=st_num[st_num>=0]##
    ##save durations of labels in each file
    label_dict={}
    start_dict={}
    for li in st_num:
        label_dict[str(li)]=[]
        start_dict[str(li)]=[]
    start=0
    for sti in range(1,len(st)):
        if sti==len(st)-1 or st[sti]!=st[start]:
            label_dict[str(st[start])].append(sti-start)
            start_dict[str(st[start])].append(start)
            start=sti
    return([label_dict,start_dict])


def get_close_segments(ratB,ratW,seg_long=15,min_distance_body_times=1):
    ci=get_near_indexes(ratB,ratW,min_distance_body_times)
    label_dict,start_dict=duration(ci)
    ld=label_dict['1.0']
    sd=start_dict['1.0']
    long_inds=np.argwhere(np.array(ld)>=seg_long)
    ldl=np.array(ld)[long_inds]##length of each segment
    sdl=np.array(sd)[long_inds]##starts of each segment
    ##get clips of 15 frames
    starts_frame15=[]
    fps=seg_long
    random.seed(1000)
    for i in range(ldl.shape[0]):
        quot=ldl[i,0]//fps
        wd_size=ldl[i,0]//quot
        ##print('len%s, quot%s, fps-%s, wd_size-%s'%(ldl[i,0],quot,fps,wd_size))
        l_start_inds=[i for i in range(0,ldl[i,0]-fps,wd_size)]
        for si in range(len(l_start_inds)):
            snum=l_start_inds[si]
            if si==(len(l_start_inds)-1):
                rst=random.randint(snum,ldl[i,0]-fps)
                starts_frame15.append(sdl[i,0]+rst)
            else:
                rst=random.randint(snum,snum+wd_size-fps+1)
                starts_frame15.append(sdl[i,0]+rst)
    return(starts_frame15)


def get_far_segments(ratB,ratW,seg_long=15,min_distance_body_times=1):
    ci=get_far_indexes(ratB,ratW,min_distance_body_times)
    label_dict,start_dict=duration(ci)
    ld=label_dict['1.0']
    sd=start_dict['1.0']
    long_inds=np.argwhere(np.array(ld)>=seg_long)
    ldl=np.array(ld)[long_inds]##length of each segment
    sdl=np.array(sd)[long_inds]##starts of each segment
    ##get clips of 15 frames
    starts_frame15=[]
    fps=seg_long
    random.seed(1000)
    for i in range(ldl.shape[0]):
        quot=ldl[i,0]//fps
        wd_size=ldl[i,0]//quot
        ##print('len%s, quot%s, fps-%s, wd_size-%s'%(ldl[i,0],quot,fps,wd_size))
        l_start_inds=[i for i in range(0,ldl[i,0]-fps,wd_size)]
        for si in range(len(l_start_inds)):
            snum=l_start_inds[si]
            if si==(len(l_start_inds)-1):
                rst=random.randint(snum,ldl[i,0]-fps)
                starts_frame15.append(sdl[i,0]+rst)
            else:
                rst=random.randint(snum,snum+wd_size-fps+1)
                starts_frame15.append(sdl[i,0]+rst)
    return(starts_frame15)



def cluster_plot3(embedding,classes,pathFig,title=None,centers=None):
    plt.figure(figsize=(10, 10),dpi=300)
    plt.scatter(x=embedding[:, 0], y=embedding[:, 1],c=classes, cmap='Spectral', s=2)
    if centers:
        plt.scatter(centers[:,0], centers[:,1], marker="v", picker=True)
    plt.gca().set_aspect('equal', 'datalim')
    ##
    #cMax=np.max(classes)+1
    #plt.colorbar()
    plt.xlabel('UMAP-PC1')
    plt.ylabel('UMAP-PC2')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(pathFig)


def plot_knn_decision_boundary(X, y, pathFig,k=1):
    #plt.close('all')
    # data setting
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # meshgrid的shape是:[第二个输入的.shape[0], 第一个输出的.shape[0]]
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # x.ravel就是flatten(). 然后np.c_就是按列排
    from sklearn.neighbors import KNeighborsClassifier
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X,y)
    z = knn.predict(np.c_[xx.ravel(), yy.ravel()])  # shape:[meshgrid两个shape相乘，2]
    z = z.reshape(xx.shape)  # [x*y] -> [x, y]. 这里z就是对应点的分类效果
    # levels显示的区域
    plt.contour(xx, yy, z,  cmap=plt.cm.Spectral)  # , cmp=plt.cm.Spectral or colors=['red']
    #plt.contour(xx, yy, z, levels=[10, 30, 50], cmap=plt.cm.brg)  # , cmp=plt.cm.Spectral or colors=['red']
    # plt.contour(xx, yy, y.reshape(xx.shape), levels=[0.5], colors=['blue'])
    #plt.show()
    plt.savefig(pathFig)



def plot_knn_decision_boundary_scatter(X, y, pathFig,k=1):
    #k=200
    plt.figure(figsize=(10, 10),dpi=300)
    #plt.close('all')
    # data setting
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # meshgrid的shape是:[第二个输入的.shape[0], 第一个输出的.shape[0]]
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # x.ravel就是flatten(). 然后np.c_就是按列排
    from sklearn.neighbors import KNeighborsClassifier
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X,y)
    xxyy=np.c_[xx.ravel(), yy.ravel()]
    z0 = knn.predict(xxyy)  # shape:[meshgrid两个shape相乘，2]
    z = z0.reshape(xx.shape)  # [x*y] -> [x, y]. 这里z就是对应点的分类效果
    zc=z+1  ##from 2 to differ from scatter
    # add mosaik
    plt.contourf(xx,yy,zc,cmap=plt.cm.Spectral)  #mosiak
    ##add contour
    cNOs=list(set(y))
    #plt.contour(xx, yy, z,levels=cNOs)  # , cmp=plt.cm.Spectral or colors=['red']
    c=plt.contour(xx, yy, zc,levels=list(set(zc.ravel())))  # , cmp=plt.cm.Spectral or colors=['red']
    #plt.clabel(c,inline=True,fontsize=15)
    
    ##add cluster number finally
    centers=np.stack([np.quantile(xxyy[np.where(z0==i)],0.5,axis=0) for i in cNOs])
    for i,c in enumerate(cNOs):
      #if c==4:continue
      plt.text(centers[i,0],centers[i,1],str(c),fontsize=15)
    #plt.text(2.5,6.8,'1',fontsize=15)
    #plt.text(-1.6,6.2,'3',fontsize=15)
    #plt.text(1,4.2,'4',fontsize=15)
    #plt.text(2.9,5.2,'4',fontsize=15)
    #plt.text(-1,9,'9',fontsize=15)
    #plt.text(-0.8,10.3,'19',fontsize=15)
    #plt.text(-3,6.5,'13',fontsize=15)
    ##scatter plot from 1
    plt.scatter(x=X[:, 0], y=X[:, 1],c=y, cmap='Spectral', s=4)
    ##add contour
    plt.gca().set_aspect('equal', 'datalim')
    ##
    cMax=np.max(y)+1
    plt.colorbar(boundaries=np.arange(1,cMax)).set_ticks(np.arange(1,cMax))
    plt.xlabel('UMAP-PC1')
    plt.ylabel('UMAP-PC2')
    plt.tight_layout()
    plt.savefig(pathFig)
    plt.close()



def plot_knn_decision_boundary_mosaik(X, y, pathFig,k=1):
    #k=200
    plt.figure(figsize=(10, 10),dpi=300)
    #plt.close('all')
    # data setting
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # meshgrid的shape是:[第二个输入的.shape[0], 第一个输出的.shape[0]]
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # x.ravel就是flatten(). 然后np.c_就是按列排
    from sklearn.neighbors import KNeighborsClassifier
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X,y)
    xxyy=np.c_[xx.ravel(), yy.ravel()]
    z0 = knn.predict(xxyy)  # shape:[meshgrid两个shape相乘，2]
    z = z0.reshape(xx.shape)  # [x*y] -> [x, y]. 这里z就是对应点的分类效果
    zc=z  ##from 2 to differ from scatter
    # add mosaik
    plt.contourf(xx,yy,zc,cmap=plt.cm.Spectral)  #mosiak
    ##add contour
    cNOs=list(set(y))
    #plt.contour(xx, yy, z,levels=cNOs)  # , cmp=plt.cm.Spectral or colors=['red']
    c=plt.contour(xx, yy, zc,levels=list(set(zc.ravel())))  # , cmp=plt.cm.Spectral or colors=['red']
    #plt.clabel(c,inline=True,fontsize=15)
    ##add cluster number finally with manual check
    centers=np.stack([np.quantile(xxyy[np.where(z0==i)],0.5,axis=0) for i in cNOs])
    cNOs_dicts={}
    for i in cNOs: #from 1
        cNOs_dicts[i]=centers[i-1]
    ##manual modification!!!
    cNOs_dicts[2][1]=5.5 #xy coordinates
    cNOs_dicts[8][1]=3.6
    cNOs_dicts[9][1]=0.5
    cNOs_dicts[13][0]=8.1
    cNOs_dicts[13][1]=3.7
    cNOs_dicts[16][0]=5.2
    cNOs_dicts[16][1]=3
    cNOs_dicts[20][1]=1.2
    cNOs_dicts[22][0]=9.2
    cNOs_dicts[31]=np.array([9.4,2.5])
    cNOs_dicts[37][1]=3.5
    cNOs_dicts[37][0]=7
    cNOs_dicts[38][1]=8.4
    #print(cNOs_dicts)
    ##
    for i,c in enumerate(cNOs):
      plt.text(cNOs_dicts[c][0],cNOs_dicts[c][1],str(c),fontsize=15)
    plt.gca().set_aspect('equal', 'datalim')
    ##
    cMax=np.max(y)+1
    #plt.colorbar(boundaries=np.arange(1,cMax)).set_ticks(np.arange(1,cMax))
    plt.xlabel('UMAP-PC1')
    plt.ylabel('UMAP-PC2')
    plt.tight_layout()
    plt.savefig(pathFig)
    plt.close()



def angle_project2_vectors(vector1,vector2):
    ang=angle_vectors(vector1,vector2)
    ## x-y plane angles
    ang_xy=angle_vectors(vector1[:2],vector2[:2])
    return(np.array([ang,ang_xy]))


def angle_projectXY_vectors(vector1,vector2):
    ang_xy=angle_vectors(vector1[:2],vector2[:2])
    return(ang_xy)




def get_DefVects_between(ratB,ratW):
    ###get defined vects
    f_num=ratB.shape[0]
    vect_num=4
    dim=3
    vects=np.zeros((f_num,vect_num,dim)) #(frame,vect_num=13,3)
    vects[:,0]=ratB[:,0]-ratB[:,4] #body->nose----
    vects[:,1]=ratB[:,0]-ratW[:,4] #bodyW->noseB
    vects[:,2]=ratW[:,0]-ratW[:,4] 
    vects[:,3]=ratW[:,0]-ratB[:,4]
    return(vects)


def get_project2_Angles_between2ratsNB(vects):
    f_num=vects.shape[0]
    angles=np.zeros((f_num,4))
    for fi in range(f_num):
        angles[fi,:2]=angle_project2_vectors(vects[fi,0],vects[fi,1])
        angles[fi,2:4]=angle_project2_vectors(vects[fi,2],vects[fi,3])
    return(angles)

def get_projectXY_Angles_between2ratsNB(vects):
    f_num=vects.shape[0]
    angles=np.zeros((f_num,2))
    for fi in range(f_num):
        angles[fi,0]=angle_projectXY_vectors(vects[fi,0],vects[fi,1])
        angles[fi,1]=angle_projectXY_vectors(vects[fi,2],vects[fi,3])
    return(angles)

def get_angles_velocity(angles,fps):
    diffAngs=np.diff(angles,axis=1)
    time_gap=1/fps
    angV=diffAngs/time_gap
    return(angV)


def get_angleXY_tailNeckNose_velocity(ratB,fps):
    neckNose=ratB[:,0,:2]-ratB[:,3,:2]
    neckTail=ratB[:,5,:2]-ratB[:,3,:2]
    fn=ratB.shape[0]
    angs=np.array([angle_vectors(neckNose[i],neckTail[i]) for i in range(fn)])
    diffAngs=np.zeros(fn)
    diffAngs[1:]=angs[1:]-angs[:-1]
    diffAngs[0]=diffAngs[1]
    time_gap=1/fps
    angV=diffAngs/time_gap
    return(angV)

###0-180
def nvectAng_3points(p123):
    p1,p2,p3=p123[0],p123[1],p123[2]
    n_vect=np.cross(p2-p1,p3-p1)
    length=np.linalg.norm(n_vect)
    #if length==0:length=1
    cosAngs_Z=np.dot(n_vect,np.array([0,0,1]))/length
    ang=math.acos(cosAngs_Z)*180/math.pi
    return(ang)

def shouderKnee_planeAng(ratB):
    fNum=ratB.shape[0]
    angs=np.zeros((fNum,1))
    planeTurns=[[6,8,12],[8,12,10],[12,10,6],[10,6,8]]
    for fi in range(fNum):
       angs[fi]=np.mean([nvectAng_3points(ratB[fi,planeTurns[0]]),nvectAng_3points(ratB[fi,planeTurns[1]]),nvectAng_3points(ratB[fi,planeTurns[2]]),nvectAng_3points(ratB[fi,planeTurns[3]])])
    return(angs)

def shouderBody_kneeBody_planeAng(ratB):
    fNum=ratB.shape[0]
    angs=np.zeros((fNum,2))
    planeTurns=[[6,8,4],[12,10,4]]
    for fi in range(fNum):
       angs[fi,0]=nvectAng_3points(ratB[fi,planeTurns[0]])
       angs[fi,1]=nvectAng_3points(ratB[fi,planeTurns[1]])
    return(angs)

def get_backPalmHeightDiff(ratB,ratW):
    ##
    palmBB=np.array([np.mean(ratB[:,[7,9],2],axis=1),ratB[:,4,2]]).T
    palmBW=np.array([np.mean(ratW[:,[7,9],2],axis=1),ratW[:,4,2]]).T
    return(np.concatenate((palmBB-palmBW,palmBB-palmBW[:,[1,0]]),axis=1))

def get_backPalmHeightDiff3(ratB,ratW):
    ##
    palmBB=np.array([np.mean(ratB[:,[7,9],2],axis=1),ratB[:,4,2]]).T
    palmBW=np.array([np.mean(ratW[:,[7,9],2],axis=1),ratW[:,4,2]]).T
    return(np.concatenate((palmBB-palmBW,palmBB-palmBW[:,[1,0]]),axis=1)[:,1:])

def get_angleXY_tailNose_between(ratB,ratW):
    fn=ratB.shape[0]
    tnB=ratB[:,0,:2]-ratB[:,5,:2]
    tnW=ratW[:,0,:2]-ratW[:,5,:2]
    angs=np.array([angle_vectors(tnB[i],tnW[i]) for i in range(fn)])
    return(angs)

#get overlap of 4 body parts
import numpy as np
from shapely.geometry import Polygon # 多边形

def Cal_area_2poly(data1,data2):
    """
    任意两个图形的相交面积的计算
    :param data1: 当前物体
    :param data2: 待比较的物体
    :return: 当前物体与待比较的物体的面积交集
    """
    poly1 = Polygon(data1).convex_hull # Polygon：多边形对象
    poly2 = Polygon(data2).convex_hull

    if not poly1.intersects(poly2):
        inter_area = 0 # 如果两多边形不相交
    else:
        inter_area = poly1.intersection(poly2).area # 相交面积
    return inter_area


def get_3parts(ratB):
    ratB=ratB[:,:,:2]   
    ###nose-earL-shoulderL-sholderR-earR
    headPoly=ratB[:,[0,1,6,8,2]]   
    skCenterL=(ratB[:,6]+ratB[:,10])/2
    skCenterR=(ratB[:,8]+ratB[:,12])/2
    shPoly=np.transpose(np.array([ratB[:,6],skCenterL,skCenterR,ratB[:,8]]),[1,0,2]) #sholderL-skCenterL-skCenterR-shoulderR
    knPoly=np.transpose(np.array([skCenterL,ratB[:,10],ratB[:,5],ratB[:,12],skCenterR]),[1,0,2]) #skCenterL-kneeL-tail-kneeR-skCenterR
    return([headPoly,shPoly,knPoly])


def get_3partsHeight(ratB):
    ratB=ratB[:,:,2]     ##frame,point_num   #####Height
    #print(ratB.shape)
    headPolyH=np.mean(ratB[:,[0,1,6,8,2]],axis=1)
    #
    skCenterL=(ratB[:,6]+ratB[:,10])/2
    skCenterR=(ratB[:,8]+ratB[:,12])/2
    shPolyH=np.mean(np.array([ratB[:,6],skCenterL,skCenterR,ratB[:,8]]),axis=0)
    knPolyH=np.mean(np.array([skCenterL,ratB[:,10],ratB[:,5],ratB[:,12],skCenterR]),axis=0)
    return(np.array([headPolyH,shPolyH,knPolyH]))

def get_overlap3parts_sign(ratB,ratW):
    fn=ratB.shape[0]
    overlaps=np.zeros((fn,3,3))
    ##3 polygons
    ratB3parts=get_3parts(ratB)
    ratW3parts=get_3parts(ratW)
    ##Mean Heights of 3 polygons
    ratB3partsH=get_3partsHeight(ratB)
    ratW3partsH=get_3partsHeight(ratW)
    for fi in range(fn):
      for i in range(3):
        for j in range(3):
          overlaps[fi,i,j]=Cal_area_2poly(ratB3parts[i][fi],ratW3parts[j][fi])*np.sign(ratB3partsH[i,fi]-ratW3partsH[j,fi])
    return(overlaps)


def get_NoseTail_zone(ratB,i0,j1,j2,length_ratio):
    #length_ratio=1 #1, nose, tail 0.5
    ##nose:0,earL:1,earR:2; #tail:5,kneeL:10,kneeR:12
    ratB=ratB[:,:,:2]
    center=ratB[:,i0]
    dist_noseEar = np.mean([np.mean(np.sqrt(np.sum((center-ratB[:,j1])**2, axis=1))),np.mean(np.sqrt(np.sum((center-ratB[:,j2])**2, axis=1)))])
    r=dist_noseEar*length_ratio
    ##get circle points
    xys=np.array([[center[:,0]+r*np.sin(ang),center[:,1]+r*np.cos(ang)] for ang in np.arange(0,2*np.pi,30*np.pi/180)])
    nosePoly=np.transpose(xys, [2,0,1]) 
    return(nosePoly)

def get_overlap_NoseTail_zone(ratB,ratW,ratio):
    fn=ratB.shape[0]
    overlaps=np.zeros((fn,3))
    ##nose zone, tail zone
    ratB2parts=[get_NoseTail_zone(ratB,0,1,2,ratio),get_NoseTail_zone(ratB,5,10,12,ratio)]
    ratW2parts=[get_NoseTail_zone(ratW,0,1,2,ratio),get_NoseTail_zone(ratW,5,10,12,ratio)]
    for fi in range(fn):
      i=0;j=0
      overlaps[fi,0]=Cal_area_2poly(ratB2parts[i][fi],ratW2parts[j][fi])
      i=0;j=1
      overlaps[fi,1]=Cal_area_2poly(ratB2parts[i][fi],ratW2parts[j][fi])
      i=1;j=0
      overlaps[fi,2]=Cal_area_2poly(ratB2parts[i][fi],ratW2parts[j][fi])
    return(overlaps)   


def get_overlap_NoseTail_zone_sign2(ratB,ratW,ratio):
    ##ratio is ratius ratio in nose-ear distance, ratio in tail-knee distance
    fn=ratB.shape[0]
    overlaps=np.zeros((fn,3))
    ##0 nose zone, 5 tail zone
    ratB2parts=[get_NoseTail_zone(ratB,0,1,2,ratio),get_NoseTail_zone(ratB,5,10,12,ratio)]
    ratW2parts=[get_NoseTail_zone(ratW,0,1,2,ratio),get_NoseTail_zone(ratW,5,10,12,ratio)]
    ##zone overlap * sign
    for fi in range(fn):
      i=0;j=0;i2=0;j2=0
      overlaps[fi,0]=Cal_area_2poly(ratB2parts[i][fi],ratW2parts[j][fi])*np.sign(ratB[fi,i2,2]-ratW[fi,j2,2])
      i=0;j=1;i2=0;j2=5
      overlaps[fi,1]=Cal_area_2poly(ratB2parts[i][fi],ratW2parts[j][fi])*np.sign(ratB[fi,i2,2]-ratW[fi,j2,2])
      i=1;j=0;i2=5;j2=0
      overlaps[fi,2]=Cal_area_2poly(ratB2parts[i][fi],ratW2parts[j][fi])*np.sign(ratB[fi,i2,2]-ratW[fi,j2,2])
    return(overlaps)


def hms2second(hms):
    h,m,s=hms.split(':')
    seconds=int(h)*60*60+int(m)*60+int(s)
    return(seconds)

from  matplotlib.pyplot import  hist

def hist_plot(data,filePath):
    data=data.reshape(-1)
    [n,bins,patches]=hist(data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(data, bins,rwidth=0.9)
    plt.title(u'data distribution')
    plt.xlabel(u'range')
    plt.ylabel(u'number')
    ##plt.show()
    fig.savefig(filePath)
    plt.close()

def hist_plot2(data,filePath,title,ylabel='Number',xlabel='Range'):
    data=data.reshape(-1)
    [n,bins,patches]=hist(data,20)
    fig = plt.figure(figsize=(9, 9),dpi=300)
    ax = fig.add_subplot(111)
    ax.hist(data, bins,rwidth=0.9)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ##plt.show()
    fig.savefig(filePath)
    plt.close()



def minmax_cut(data):
    pts=[0,0.1,99.9,100]
    np.random.seed(100)
    for i in range(data.shape[1]):
        data_i=data[:,i]
        m0,m01,m999,m1=np.percentile(data_i,pts)
        ##sd of clean data
        sd=np.std(data_i[(data_i>=m01)&(data_i<=m999)])
        print('%s_1:  m0:%.2f, m01:%.2f, m999:%.2f, m1:%.2f sd:%.2f'%(i,m0,m01,m999,m1,sd))
        ##filtering
        min_inds=np.argwhere(data[:,i]<m01)[:,0]
        data[min_inds,i]=m01+sd*0.1*(np.random.rand(len(min_inds))-0.5)
        max_inds=np.argwhere(data[:,i]>m999)[:,0]
        data[max_inds,i]=m999+sd*0.1*(np.random.rand(len(max_inds))-0.5)
        ##-1,1 normalize
        m0,m01,m999,m1=np.percentile(data[:,i],pts)
        print('%s_2:  m0:%.2f, m01:%.2f, m999:%.2f, m1:%.2f'%(i,m0,m01,m999,m1))
        data[:,i]=2*(data[:,i]-m01)/(m999-m01)-1
    return(data)


def minmax_cutNN(data):
    pts=[0,0.1,99.9,100]
    np.random.seed(100)
    for i in range(data.shape[1]):
        data_i=data[:,i]
        m0,m01,m999,m1=np.percentile(data_i,pts)
        ##sd of clean data
        sd=np.std(data_i[(data_i>=m01)&(data_i<=m999)])
        print('%s_1:  m0:%.2f, m01:%.2f, m999:%.2f, m1:%.2f sd:%.2f'%(i,m0,m01,m999,m1,sd))
        ##filling outliers
        min_inds=np.argwhere(data[:,i]<m01)[:,0]
        data[min_inds,i]=m01+sd*0.1*(np.random.rand(len(min_inds))-0.5)
        max_inds=np.argwhere(data[:,i]>m999)[:,0]
        data[max_inds,i]=m999+sd*0.1*(np.random.rand(len(max_inds))-0.5)
        ##-1,1 normalize
        m0,m01,m999,m1=np.percentile(data[:,i],pts)
        print('%s_2:  m0:%.2f, m01:%.2f, m999:%.2f, m1:%.2f'%(i,m0,m01,m999,m1))
        data[:,i]=(data[:,i]-m01)/(m999-m01)
    return(data)

##return minmax parameters
def minmax_cut2(data):
    pts=[0,0.1,99.9,100]
    m19sd19=np.zeros((5,data.shape[1]))
    for i in range(data.shape[1]):
        data_i=data[:,i]
        m0,m011,m9991,m1=np.percentile(data_i,pts)
        ##sd of clean data
        sd=np.std(data_i[(data_i>=m011)&(data_i<=m9991)])
        print('%s_1:  m0:%.2f, m011:%.2f, m9991:%.2f, m1:%.2f sd:%.2f'%(i,m0,m011,m9991,m1,sd))
        ##filling outliers
        min_inds=np.argwhere(data[:,i]<m011)[:,0]
        np.random.seed(i*123)
        data[min_inds,i]=m011+sd*0.1*(np.random.rand(len(min_inds))-0.5)
        max_inds=np.argwhere(data[:,i]>m9991)[:,0]
        np.random.seed(i*321)
        data[max_inds,i]=m9991+sd*0.1*(np.random.rand(len(max_inds))-0.5)
        ##-1,1 normalize
        m0,m012,m9992,m1=np.percentile(data[:,i],pts)
        print('%s_2:  m0:%.2f, m012:%.2f, m9992:%.2f, m1:%.2f'%(i,m0,m012,m9992,m1))
        data[:,i]=2*(data[:,i]-m012)/(m9992-m012)-1
        #
        m19sd19[:,i]=np.array([m011,m9991,sd,m012,m9992])
    return([data,m19sd19])

###apply minmax to new data
def minmax_cut_apply(data,m19sd19):
    for i in range(data.shape[1]):
        m011,m9991,sd,m012,m9992=m19sd19[:,i]
        ##filling outliers
        min_inds=np.argwhere(data[:,i]<m011)[:,0]
        np.random.seed(i*123)
        data[min_inds,i]=m011+sd*0.1*(np.random.rand(len(min_inds))-0.5)
        max_inds=np.argwhere(data[:,i]>m9991)[:,0]
        np.random.seed(i*321)
        data[max_inds,i]=m9991+sd*0.1*(np.random.rand(len(max_inds))-0.5)
        ##-1,1 normalize
        data[:,i]=2*(data[:,i]-m012)/(m9992-m012)-1
    return(data)


def discretize3_2_half(data):
    pts=[20,60]
    np.random.seed(100)
    for i in range(data.shape[1]):
        data_i=data[:,i]
        m33,m66=np.percentile(data_i,pts)
        data_i[data_i<m33]=0
        #
        data[:,i]=data_i
    return(data)

def get_overlap_wholeBody(ratB,ratW):
    fn=ratB.shape[0]
    overlaps=np.zeros((fn,1))
    ##body poly,##nose,earL,shoulderL,kneeL,tail,kneeR,shoulderR.earR
    bodyInds=[0,1,6,10,5,12,8,2]
    bodyB=ratB[:,bodyInds,:2]
    bodyW=ratW[:,bodyInds,:2]
    for fi in range(fn):
        overlaps[fi]=Cal_area_2poly(bodyB[fi],bodyW[fi])
    return(overlaps)


def get_behavior_byFrame(bhvEachS,sfsEachS,realStart,realEnd):  ##bhvEachS corresponded to sfs
    ##get used video segment
    #all use relative start 0
    behavior=np.zeros(realEnd-realStart)
    sfsEachS=[i-realStart for i in sfsEachS]
    ##calculate transition by frame
    for si,s in enumerate(sfsEachS):behavior[s:s+seg_length]=bhvEachS[si]
    ##fill gaps
    for si in range(len(sfsEachS)-1):
       gap=sfsEachS[si+1]-sfsEachS[si]-seg_length
       if gap<gapTh and gap>0:
         ##print(gap)
         behavior[sfsEachS[si]+seg_length:sfsEachS[si+1]]=bhvEachS[si]#fill by early behavior
    ##fill fs header and tail
    if sfsEachS[0]<=fs:behavior[:fs]=bhvEachS[0]
    if sfsEachS[-1]+seg_length>=len(behavior)-fs:behavior[-fs:]=bhvEachS[-1]
    return(behavior)




##Simple features of each rat
fps=30
# %%
#curve fitting
def exp_decay(x,k,p):return (1-p)*np.exp(-k *x)+p  ##suitable for correlation start from 1
from scipy.optimize import curve_fit
def get_acr_tau(resPath,fname):
  acrs=np.load(join(resPath,'autocorr-%sseconds.npz'%(duration)))[fname]
  ##x2=xx  ##bad fitting
  x2=[i/1000 for i in xx]
  ##fit auto-correlation of each feature
  taus=[]
  popts=[]
  for i in range(acrs.shape[0]):
    popt, _ = curve_fit(exp_decay,x2,acrs[i],method='lm')
    popts.append(popt)
    taus.append(1/popt[0])
  ##all features; tau=1.08+-0.83, duration=60
  tau_median=np.median(taus)
  tau_mean=np.mean(taus)
  tau_sd=np.std(taus)
  ##
  np.savez(join(resPath,'%s-eachFeat_autocorrZSPCA-%sseconds-taus_median%s-mean%s-sd%s'%(fname,duration,
          round(tau_median,2),round(tau_mean,2),round(tau_sd,2))),tau_median=tau_median,tau_mean=tau_mean,tau_sd=tau_sd,taus=taus)
  ##
  resPathFit=join(resPath,'%s-autocorrFit'%(fname))
  os.makedirs(resPathFit,exist_ok=True)
  ##
  for j in range(acrs.shape[0]):
    popt=popts[j]
    y2 = [exp_decay(i,popt[0],popt[1]) for i in x2]
    plt.plot(x2,y2,'r--')
    plt.plot(x2,acrs[j],'b--')
    #plt.show()
    plt.savefig(join(resPathFit,'feat_'+str(j)+'acrfit.pdf'))
    plt.close()


def bhv_start_end(bseq,bs):
  starts=[]
  ends=[]
  for bi in range(len(bseq)):
    if bi==0 and bseq[bi]==bs:
        starts.append(bi)
    if bi>0  and bseq[bi-1]!=bs and bseq[bi]==bs:
        starts.append(bi)
    if bi<len(bseq)-1 and bseq[bi]==bs and bseq[bi+1]!=bs:
        ends.append(bi+1)
    if bi==len(bseq)-1 and bseq[bi]==bs:
        ends.append(bi+1)
  return(np.array([starts,ends]))



#perc=50  #10,50 70
def get_startF_ByLabelProb(lpv_black,perc):
  rowStep=9  #10 Hz
  midFrame=rowStep//2
  ##
  startFrames=[]
  labels=[]
  indsL=[lis+1 for lis in range(lpv_black.shape[0]-1) if lpv_black['startFrame'].values[lis+1]-lpv_black['startFrame'].values[lis]>3] 
  ##-0.4,0,0.4;9 frames in 10 Hz, by frame NO in 30 fps
  indsL.insert(0,0);indsL.append(lpv_black.shape[0])
  for ii in range(len(indsL)-1):
    ##data used 
    lp0=lpv_black.iloc[indsL[ii]:indsL[ii+1]]
    #
    bseq=lp0['cluster_labels'].values
    ### label of more currence in half later
    labelCheck=np.argmax(np.bincount(bseq[midFrame:]))
    ##choose the longest segment of the checked label
    se=bhv_start_end(bseq,labelCheck)
    indsSE=se[:,np.argmax(se[1]-se[0])]
    inds=np.arange(indsSE[0],indsSE[1])
    #start frame by prob percentile
    probUse=lp0['cluster_pvalues'].values[inds]
    frameNO=inds[np.where(probUse>=np.percentile(probUse,perc))[0][0]]  #perc in each range or for each behavior class ?
    ##start frame NO in the video
    startFrames.append(lp0['startFrame'].values[frameNO]) 
    labels.append(labelCheck)
  return(pd.DataFrame({'label':labels,'startFrame10Hz':startFrames}))

def get_startF_ByLabelProb2(lpv_black,perc):
  rowStep=9  #10 Hz
  midFrame=rowStep//2
  #perc=50  #10,50 70
  ##
  startFrames=[]
  labels=[]
  frameNOs=[]
  probs=[]
  #
  indsL=[lis+1 for lis in range(lpv_black.shape[0]-1) if lpv_black['startFrame'].values[lis+1]-lpv_black['startFrame'].values[lis]>3] 
  ##-0.4,0,0.4;9 frames in 10 Hz, by frame NO in 30 fps
  indsL.insert(0,0);indsL.append(lpv_black.shape[0])
  for ii in range(len(indsL)-1):
    ##data used 
    lp0=lpv_black.iloc[indsL[ii]:indsL[ii+1]]
    #
    bseq=lp0['cluster_labels'].values
    ### label of more currence in half later
    #
    #local index
    labelCheck=np.argmax(np.bincount(bseq[midFrame:]))
    #probability
    lp0['cluster_pvalues'][lp0['cluster_labels']!=labelCheck]=0
    probs.append(lp0['cluster_pvalues'].values)
    ##choose the longest segment of the checked label
    se=bhv_start_end(bseq,labelCheck)
    indsSE=se[:,np.argmax(se[1]-se[0])]
    inds=np.arange(indsSE[0],indsSE[1])
    #start frame by prob percentile
    probUse=lp0['cluster_pvalues'].values[inds]
    frameNO=inds[np.where(probUse>=np.percentile(probUse,perc))[0][0]]  #perc in each range or for each behavior class ?
    frameNOs.append(frameNO)##for probs
    #
    #global index
    #start frame NO in the video
    startFrames.append(lp0['startFrame'].values[frameNO]) 
    labels.append(labelCheck)
  return([pd.DataFrame({'label':labels,'startFrame10Hz':startFrames}),probs,frameNOs])


#perc=50  #10,50 70
def get_startF_ByLabelProb_preBhv(lpv_black,perc,bseqFull):
  rowStep=9  #10 Hz
  midFrame=rowStep//2
  ##
  startFrames=[]
  labels=[]
  preBhvs=[]
  #
  indsL=[lis+1 for lis in range(lpv_black.shape[0]-1) if lpv_black['startFrame'].values[lis+1]-lpv_black['startFrame'].values[lis]>3] 
  ##-0.4,0,0.4;9 frames in 10 Hz, by frame NO in 30 fps
  indsL.insert(0,0);indsL.append(lpv_black.shape[0])
  for ii in range(len(indsL)-1):
    ##data used 
    lp0=lpv_black.iloc[indsL[ii]:indsL[ii+1]]
    preBhvs.append()
    ##
    indNow=lpv_black['startFrame'].values[indsL[ii]]
    bseqFull[indNow]
    #
    bseq=lp0['cluster_labels'].values
    ### label of more currence in half later
    labelCheck=np.argmax(np.bincount(bseq[midFrame:]))
    ##choose the longest segment of the checked label
    se=bhv_start_end(bseq,labelCheck)
    indsSE=se[:,np.argmax(se[1]-se[0])]
    inds=np.arange(indsSE[0],indsSE[1])
    #start frame by prob percentile
    probUse=lp0['cluster_pvalues'].values[inds]
    frameNO=inds[np.where(probUse>=np.percentile(probUse,perc))[0][0]]  #perc in each range or for each behavior class ?
    ##start frame NO in the video
    startFrames.append(lp0['startFrame'].values[frameNO]) 
    labels.append(labelCheck)
  return(pd.DataFrame({'label':labels,'startFrame10Hz':startFrames}))




def plot_bootstrap_matrix(
    x, y, ci=99, label=None, color=None, alpha=0.5, mu=None, linewidth=1, ax=None, **plt_kwargs
):
    #Expects a tidy dataframe that has already been bootstrapped. I.e., each trial is the
    #average of a different sample (with replacement) of the original data
    if mu is None:
        _mu = np.nanmean(y, axis=0)
        mu = _mu
    if ci == "sd":
        lo = mu - 2 * np.nanstd(y, axis=0)
        hi = mu + 2 * np.nanstd(y, axis=0)
    elif ci=='1sd':
        lo = mu - np.nanstd(y, axis=0)
        hi = mu + np.nanstd(y, axis=0)
    elif ci=='1se':
        lo = mu - np.nanstd(y, axis=0)/y.shape[0]**0.5
        hi = mu + np.nanstd(y, axis=0)/y.shape[0]**0.5
    else:
        lo = np.nanquantile(y, 1 - (ci / 100), axis=0)
        hi = np.nanquantile(y, ci / 100, axis=0)

    plotter = plt if ax is None else ax

    lines = plotter.plot(x, mu, label=label, color=color, linewidth=linewidth, **plt_kwargs)
    color = lines[-1].get_color()
    plotter.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0, **plt_kwargs)

    return plt.gca() if ax is None else ax

##smooth data
def moving_average(interval, windowsize):
   window = np.ones(int(windowsize))/float(windowsize)
   re = np.convolve(interval,window,'same')
   return re


##calculate AUC of 2 lines
from scipy.integrate import trapz
##calculate AUC 
def get_AUC(exp_low,con_low):
    ##exp_low,con_low: 1D numpy array
    ##calculate AUC, experiment and control data
    auc = get_AUC_sign(exp_low,con_low) #-1,1
    auc = (auc+1)/2 #0,1
    return(auc)

def get_AUC_sign(exp_low,con_low):
    ##exp_low,con_low: 1D numpy array
    ##calculate AUC, experiment and control data
    #auc_l = []
    #for i in range(41):
    #exp_low, con_low = exp_low_dimen[:, i], con_low_dimen[:, i]
    exp_low = exp_low + np.random.normal(0, 0.001, exp_low.shape)
    con_low = con_low + np.random.normal(0, 0.001, con_low.shape)
    per_exp = np.percentile(exp_low, np.arange(0, 100, 1),
            interpolation='midpoint').ravel()
    per_con = np.percentile(con_low, np.arange(0, 100, 1),
            interpolation='midpoint').ravel()
    xlimmin = min(per_exp.min(), per_con.min())
    xlimmax = max(per_exp.max(), per_con.max())
    xticks = np.linspace(xlimmin, xlimmax, 100)
    y_exp = np.interp(xticks, per_exp, np.arange(0, 100, 1))
    y_con = np.interp(xticks, per_con, np.arange(0, 100, 1))
    auc = -(trapz(y_exp, y_con) - 5000)/5000  #-1,1
    return(auc)

##calculate AUC and P value
def get_AUC_P(exp_low,con_low):
    #exp_low=np.array([2,3,4,3,4,3,4,3,4])
    #con_low=np.array([0,9,8,7,6,5])
    ##real AUC
    auc=get_AUC(exp_low,con_low)
    ##shuffled AUCs
    N_ITER=5000
    expN=len(exp_low)
    dataAll=np.concatenate([exp_low,con_low],axis=0)
    ##
    shuff_aucs=np.zeros(N_ITER)
    for i in range(N_ITER):
        random.shuffle(dataAll)
        shuff_aucs[i]=get_AUC(dataAll[:expN],dataAll[expN:])
    pval = np.sum(abs(shuff_aucs-0.5) > abs(auc-0.5)) / N_ITER
    ##
    #import matplotlib.pyplot as plt
    #plt.hist(shuff_aucs, bins=30, color='skyblue', alpha=0.8)
    return([auc,pval])
    
def get_P_star_color(tP):
    if tP<0.001:
        star='***';color='red'
    elif tP<0.01:
        star='**';color='red'
    elif tP<0.05:
        star='*';color='red'
    else:
        star=str(round(tP,2));color='black'
    return([star,color])

##get event: chase continous 0.8s ,other 1.6s 
def get_event(bseqFull,bhvTurn,fast_duration):
    lb_st10Hz=[]
    Hz=10
    for bhvNO,bhvName in bhvTurn.items():
        ses=np.int64(bhv_start_end(bseqFull,bhvNO))
        if bhvNO in fast_duration.keys():  ##chase 0.4
          ses_long=ses[:,ses[1]-ses[0]>=fast_duration[bhvNO]*Hz]
        else:  #other 1.6
          ses_long=ses[:,ses[1]-ses[0]>=fast_duration['Other']*Hz]
        ##kepp onset around 1.6s
        ses_long=ses_long[:,(ses_long[0]>=1.6*Hz) & (ses_long[0]<=len(bseqFull)-1.6*Hz)]
        #previous 0.6s most label as preBhv
        pre_bhvLabel=np.int64([np.argmax(np.bincount(bseqFull[startFrame-6:startFrame])) for startFrame in ses_long[0]])
        pre_bhvName=[bhvTurn[num] for num in pre_bhvLabel]
        bhv_df=pd.DataFrame({'startFrame10Hz':ses_long[0],'endFrame10Hz':ses_long[1],'bhvLabel':bhvNO,'bhvName':bhvName,
                      'pre_bhvLabel':pre_bhvLabel,'pre_bhvName':pre_bhvName,})
        lb_st10Hz.append(bhv_df)
    lb_st10Hz=pd.concat(lb_st10Hz)
    return(lb_st10Hz)

##get event: chase continous 0.8s ,other 1.6s 
def get_event_usese(bseqFull,bhvTurn,fast_duration):
    lb_st10Hz=[]
    Hz=10
    for bhvNO,bhvName in bhvTurn.items():
        ses=np.int64(bhv_start_end(bseqFull,bhvNO))
        if bhvNO in fast_duration.keys():  ##chase 0.4
          ses_long=ses[:,ses[1]-ses[0]>=fast_duration[bhvNO]*Hz]
        else:  #other 1.6
          ses_long=ses[:,ses[1]-ses[0]>=fast_duration['Other']*Hz]
        bhv_df=pd.DataFrame({'startFrame10Hz':ses_long[0],'endFrame10Hz':ses_long[1],'bhvLabel':bhvNO,'bhvName':bhvName})
        lb_st10Hz.append(bhv_df)
    lb_st10Hz=pd.concat(lb_st10Hz)
    return(lb_st10Hz)

####feature used for behavior event filteration
bhv_feat={'1_Rear':'1_backH','2_Leave':'12_dist_backback','4_W-approached':'v2_conv', '5_Sniffed':'2_1_sniffAno_abs', '6_SlowChased':'v2_conv', '7_FastChased':'v2_conv', 
        '8_Pounced':'1_body-2_body_abs', '9_Pinned':'1_body-2_body_abs',  '10_W-approach':'v1_conv','11_Sniff':'1_2_sniffAno_abs','12_SlowChase':'v1_conv', 
        '13_FastChase':'v1_conv','14_Pounce':'1_body-2_body_abs', '15_Pin':'1_body-2_body_abs'}
def get_lab_startEnd2sNRC(df,bhv_feat,bhvDicts,csd):
    Hz=10
    ##feature SD thresholds for event filtering
    #csd=0.4
    featSD0={'v1_conv':csd,'v2_conv':csd,'1_2_sniffAno_abs':0.5,'2_1_sniffAno_abs':0.5,'1_body-2_body_abs':0.5,'1_backH':1.0,'12_dist_backback':1.0} 
    featSDs={'v1_conv':csd,'v2_conv':csd,'Other':0.5};preF=5
    other=['1_Rear','2_Leave']
    snPlay=['5_Sniffed','8_Pounced','9_Pinned','11_Sniff','14_Pounce','15_Pin'] ##2 thresholds pre
    chase=['10_W-approach','12_SlowChase', '13_FastChase','4_W-approached','6_SlowChased', '7_FastChased']
    ##
    df['1_2_sniffAno_abs']=df['1_nose-2_tail_absL']
    df['2_1_sniffAno_abs']=df['1_tail-2_nose_absL']
    ##smooth,##sniff-abs and body-body abs smooth
    df['v1_conv']=np.convolve(df['v1'],np.ones(Hz),mode='same')
    df['v2_conv']=np.convolve(df['v2'],np.ones(Hz),mode='same')
    df['1_2_sniffAno_abs']=np.convolve(df['1_2_sniffAno_abs'],np.ones(Hz//2),mode='same')
    df['2_1_sniffAno_abs']=np.convolve(df['2_1_sniffAno_abs'],np.ones(Hz//2),mode='same')
    df['1_body-2_body_abs']=np.convolve(df['1_body-2_body_abs'],np.ones(Hz//2),mode='same')
    feats=['v1_conv','v2_conv','1_backH','12_dist_backback','1_2_sniffAno_abs','2_1_sniffAno_abs','1_body-2_body_abs']
    df3=df[['bhvName', 'bhvNO']+feats]  #, 'photZS_ori'
    #######Behavior event filteration by duration
    #bseq=df3['bhvNO'].values
    fast_duration={2:0.8,3:0.8,5:0.8,6:0.8,10:0.8,11:0.8,'Other':1.0};#durTh='0.81.6' #'0.81.0' pounce not increase much
    lb_st10Hz=get_event_usese(df3['bhvNO'].values,bhvDicts,fast_duration)
    lb_st10Hz['dur']=lb_st10Hz['endFrame10Hz']-lb_st10Hz['startFrame10Hz']
    #
    ########Behavior event filteration by feature value####
    featThreshs={}
    featThreshsM={}
    featThreshsSD={}
    for featN in featSD0.keys():
        feat=df3[featN]
        featThreshs[featN]=featSD0[featN]*np.std(feat) if featN=='12_dist_backback' else np.mean(feat)+featSD0[featN]*np.std(feat)##distance change range for leave
        featThreshsM[featN]=np.mean(feat)
        featSD=featSDs.get(featN,featSDs['Other'])
        featThreshsSD[featN]=np.mean(feat)+featSD*np.std(feat)
    #feature value > Threshold, time points
    bhv_threshUp_duration=2   
    ####bhv event filteration by feature 
    lb_st10Hz_ff=[]
    for li in range(lb_st10Hz.shape[0]):
        lbs=lb_st10Hz.iloc[li]
        bName=lbs['bhvName']
        if bName not in bhv_feat.keys():continue   
        featName=bhv_feat[bName]
        feat=df3[featName].values[lbs['startFrame10Hz']:lbs['endFrame10Hz']]
        if bName=='2_Leave':#2_leave: dist increase value>range
            if np.max(feat)-np.min(feat)>featThreshs[featName]:lb_st10Hz_ff.append(lbs)
        elif  bName=='5_Sniffed': ##no rear
            featName2='1_backH'
            feat2=df3[featName2].values[lbs['startFrame10Hz']:lbs['endFrame10Hz']]
            if (sum(feat>=featThreshs[featName]) >=bhv_threshUp_duration) and (sum(feat2>=featThreshs[featName2])==0):lb_st10Hz_ff.append(lbs) 
        else:
            if sum(feat>=featThreshs[featName]) >=bhv_threshUp_duration:lb_st10Hz_ff.append(lbs)
    if len(lb_st10Hz_ff)==0:return([])
    lb_st10Hz_ff=pd.concat(lb_st10Hz_ff,axis=1).T 
    #
    #########Behavior onset refinement by feature peaks or turning points or minimum
    lb_st10Hz_ff2=[]
    for li in range(lb_st10Hz_ff.shape[0]):
        lb_info=lb_st10Hz_ff.iloc[li]
        bhvN=lb_info['bhvName'];featN=bhv_feat[bhvN]
        ##get feature values
        sf0=lb_info['startFrame10Hz'];sf=max(0,sf0-preF);ef=lb_info['endFrame10Hz']
        featUse=df3[bhv_feat[bhvN]].values[sf:ef]
        if bhvN in snPlay:compLen=10;foldM=1.5
        if bhvN in chase:compLen=5;foldM=1.0    
        if bhvN in other:compLen=10;foldM=1.0
        if bhvN in other: #label around minimum, ok for rear and leave
            peaksF_der2=np.argmin(df3[bhv_feat[bhvN]].values[sf:(sf0+preF+1)])
        else: ###2 thresholds pre
            peakF=np.min(np.where(featUse>=featThreshsSD[featN])[0])
            peaksF_der2=np.max(np.where(featUse[:(peakF+1)]>=featThreshsM[featN])[0])
        if np.mean(featUse[peaksF_der2:min((peaksF_der2+compLen),len(featUse))])>foldM*np.mean(featUse[max(0,peaksF_der2-compLen):peaksF_der2]):#later>pre, filter events
            lb_info['startFrame10Hz']=sf+peaksF_der2
            lb_st10Hz_ff2.append(lb_info)
    if len(lb_st10Hz_ff2)>0:
            lb_st10Hz_ff=pd.concat(lb_st10Hz_ff2,axis=1).T 
            return(lb_st10Hz_ff)
    else:
        return([])
    return(lb_st10Hz_ff)


def get_lab_startEnd10sNRC(df,bhv_feat,bhvDicts,csd):
    Hz=10
    ##feature SD thresholds for event filtering
    featSD0={'v1_conv':csd,'v2_conv':csd,'1_2_sniffAno_abs':0.5,'2_1_sniffAno_abs':0.5,'1_body-2_body_abs':0.5,'1_backH':1.0,'12_dist_backback':1.0} 
    featSDs={'v1_conv':csd,'v2_conv':csd,'Other':0.5};preF=5
    other=['1_Rear','2_Leave']
    ##
    df['1_2_sniffAno_abs']=df['1_nose-2_tail_absL']
    df['2_1_sniffAno_abs']=df['1_tail-2_nose_absL']
    ##smooth,##sniff-abs and body-body abs smooth
    df['v1_conv']=np.convolve(df['v1'],np.ones(Hz),mode='same')
    df['v2_conv']=np.convolve(df['v2'],np.ones(Hz),mode='same')
    df['1_2_sniffAno_abs']=np.convolve(df['1_2_sniffAno_abs'],np.ones(Hz//2),mode='same')
    df['2_1_sniffAno_abs']=np.convolve(df['2_1_sniffAno_abs'],np.ones(Hz//2),mode='same')
    df['1_body-2_body_abs']=np.convolve(df['1_body-2_body_abs'],np.ones(Hz//2),mode='same')
    feats=['v1_conv','v2_conv','1_backH','12_dist_backback','1_2_sniffAno_abs','2_1_sniffAno_abs','1_body-2_body_abs']
    df3=df[['bhvName', 'bhvNO']+feats]  #, 'photZS_ori'
    #######Behavior event filteration by duration
    #bseq=df3['bhvNO'].values
    fast_duration={2:0.8,3:0.8,5:0.8,6:0.8,10:0.8,11:0.8,'Other':1.0};#durTh='0.81.6' #'0.81.0' pounce not increase much
    lb_st10Hz=get_event_usese(df3['bhvNO'].values,bhvDicts,fast_duration)
    lb_st10Hz['dur']=lb_st10Hz['endFrame10Hz']-lb_st10Hz['startFrame10Hz']
    #
    ########Behavior event filteration by feature value####
    featThreshs={}
    featThreshsM={}
    featThreshsSD={}
    for featN in featSD0.keys():
        feat=df3[featN]
        featThreshs[featN]=featSD0[featN]*np.std(feat) if featN=='12_dist_backback' else np.mean(feat)+featSD0[featN]*np.std(feat)##distance change range for leave
        featThreshsM[featN]=np.mean(feat)
        featSD=featSDs.get(featN,featSDs['Other'])
        featThreshsSD[featN]=np.mean(feat)+featSD*np.std(feat)
    #feature value > Threshold, time points
    bhv_threshUp_duration=1  #1   
    ####bhv event filteration by feature 
    lb_st10Hz_ff=[]
    for li in range(lb_st10Hz.shape[0]):
        lbs=lb_st10Hz.iloc[li]
        bName=lbs['bhvName']
        if bName not in bhv_feat.keys():continue
        featName=bhv_feat[bName]
        feat=df3[featName].values[lbs['startFrame10Hz']:lbs['endFrame10Hz']]
        if bName=='2_Leave':#2_leave: dist increase value>range
            if np.max(feat)-np.min(feat)>featThreshs[featName]:lb_st10Hz_ff.append(lbs)
        elif  bName=='5_Sniffed': ##no rear
            featName2='1_backH'
            feat2=df3[featName2].values[lbs['startFrame10Hz']:lbs['endFrame10Hz']]
            if (sum(feat>=featThreshs[featName]) >=bhv_threshUp_duration) and (sum(feat2>=featThreshs[featName2])==0):lb_st10Hz_ff.append(lbs) 
        else:
            if sum(feat>=featThreshs[featName]) >=bhv_threshUp_duration:lb_st10Hz_ff.append(lbs)
    if len(lb_st10Hz_ff)==0:return([])
    lb_st10Hz_ff=pd.concat(lb_st10Hz_ff,axis=1).T 
    #
    #########Behavior onset refinement by feature peaks or turning points or minimum
    lb_st10Hz_ff2=[]
    for li in range(lb_st10Hz_ff.shape[0]):
        lb_info=lb_st10Hz_ff.iloc[li]
        bhvN=lb_info['bhvName'];featN=bhv_feat[bhvN]
        ##get feature values
        sf0=lb_info['startFrame10Hz'];sf=max(0,sf0-preF);ef=lb_info['endFrame10Hz']
        featUse=df3[bhv_feat[bhvN]].values[sf:ef]
        compLen_pre=5;compLen_lat=10
        foldM=1.0
        if bhvN in other: #label around minimum, ok for rear and leave
            peaksF_der2=np.argmin(df3[bhv_feat[bhvN]].values[sf:(sf0+preF+1)])
        else: ###2 thresholds pre
            peakF=np.min(np.where(featUse>=featThreshsSD[featN])[0])
            peaksF_der2=np.max(np.where(featUse[:(peakF+1)]>=featThreshsM[featN])[0])
        if np.mean(featUse[peaksF_der2:min((peaksF_der2+compLen_lat),len(featUse))])>foldM*np.mean(featUse[max(0,peaksF_der2-compLen_pre):peaksF_der2]):#later>pre, filter events
            lb_info['startFrame10Hz']=sf+peaksF_der2
            lb_st10Hz_ff2.append(lb_info)
    if len(lb_st10Hz_ff2)>0:
            lb_st10Hz_ff=pd.concat(lb_st10Hz_ff2,axis=1).T 
            return(lb_st10Hz_ff)
    else:
        return([])

    return(lb_st10Hz_ff)
