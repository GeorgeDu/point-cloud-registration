### Point Cloud Registration: Papers and Codes

Point cloud registration means aligning pairs of point clouds that lie in different positions and orientations, which contains **global registration** and **local registration**. **Global registration** means coarse registration, where the pairs of point clouds have large transformations, and global registration provides an initial alignment. **Local registration** means fine registration, where the poses of the point clouds have little difference. 



#### 0. Survey papers:

***2020:***

**[arXiv]** When Deep Learning Meets Data Alignment: A Review on Deep Registration Networks (DRNs), [[paper](https://arxiv.org/pdf/2003.03167.pdf)]

**[arXiv]** Least Squares Optimization: from Theory to Practice, [[paper](https://arxiv.org/pdf/2002.11051.pdf)]

***2012:***

**[TVCG]** Registration of 3D Point Clouds and Meshes: A Survey From Rigid to Non-Rigid, [[paper](http://orca.cf.ac.uk/47333/1/ROSIN registration of 3d point clouds and meshes.pdf)]



#### 1. Global Registration

Most of the global registration methods operate on candidate correspondences.  Other approaches are based on the branch-and-bound techniques, which explore the pose space exhaustively. 

#### 1.1 Finding correspondences

***2020:***

**[ECCV]** DH3D: Deep Hierarchical 3D Descriptors for Robust Large-Scale 6DoF Relocalization, [[paper](https://arxiv.org/pdf/2007.09217.pdf)]

**[ECCV]** Iterative Distance-Aware Similarity Matrix Convolution with Mutual-Supervised Point Elimination for Efficient Point Cloud Registration, [[paper](https://arxiv.org/abs/1910.10328)] [[code](https://github.com/jiahaowork/idam)]

**[PRL]** Fuzzy Logic and Histogram of Normal Orientation-based 3D Keypoint Detection for Point Clouds, [[paper](https://www.sciencedirect.com/science/article/abs/pii/S016786552030180X)]

**[arXiv]** Learning 3D-3D Correspondences for One-shot Partial-to-partial Registration, [[paper](https://arxiv.org/pdf/2006.04523.pdf)]

**[arXiv]** RPM-Net: Robust Point Matching using Learned Features, [[paper](https://arxiv.org/pdf/2003.13479.pdf)]

**[arXiv]** End-to-End Learning Local Multi-view Descriptors for 3D Point Clouds, [[paper](https://arxiv.org/pdf/2003.05855.pdf)]

**[arXiv]** D3Feat: Joint Learning of Dense Detection and Description of 3D Local Features, [[paper](https://arxiv.org/pdf/2003.03164.pdf)]

**[arXiv]** Self-supervised Point Set Local Descriptors for Point Cloud Registration, [[paper](https://arxiv.org/pdf/2003.05199.pdf)]

**[arXiv]** StickyPillars: Robust feature matching on point clouds using Graph Neural Networks, [[paper](https://arxiv.org/pdf/2002.03983.pdf)]

**[arXiv]** LRF-Net: Learning Local Reference Frames for 3D Local Shape Description and Matching, [[paper](https://arxiv.org/pdf/2001.07832.pdf)]

***2019:***

**[CVPR]** 3DRegNet: A Deep Neural Network for 3D Point Registration, [[paper](https://arxiv.org/abs/1904.01701)] [[code](https://github.com/goncalo120/3DRegNet)]

**[ICCV]** DeepICP: An End-to-End Deep Neural Network for 3D Point Cloud Registration, [[paper](https://arxiv.org/pdf/1905.04153v2.pdf)]

**[ICCV]** Deep Closest Point: Learning Representations for Point Cloud Registration, [[paper](https://arxiv.org/abs/1905.03304)] [[code](https://github.com/WangYueFt/dcp)]

**[CVPR]** The Perfect Match: 3D Point Cloud Matching with Smoothed Densities, [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Gojcic_The_Perfect_Match_3D_Point_Cloud_Matching_With_Smoothed_Densities_CVPR_2019_paper.pdf)]

***2018:***

**[ECCV]** 3DFeat-Net: Weakly Supervised Local 3D Features for Point Cloud Registration, [[paper](https://arxiv.org/abs/1807.09413)] [[code](https://github.com/yewzijian/3DFeatNet)]

***2017:***

**[CVPR]** 3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions, [[paper](https://arxiv.org/abs/1603.08182)] [[code](https://github.com/andyzeng/3dmatch-toolbox)]

***2016:***

**[ECCV]** Fast Global Registration, [[paper](http://vladlen.info/papers/fast-global-registration.pdf)] [[code](https://github.com/intel-isl/FastGlobalRegistration)]

**[IJCV]** A comprehensive performance evaluation of 3D local feature descriptors, [[paper](https://link.springer.com/article/10.1007/s11263-015-0824-y)]

***2014:***

**[CVIU]** SHOT: Unique signatures of histograms for surface and texture description, [[paper](https://www.sciencedirect.com/science/article/pii/S1077314214000988))]

**[SGP]** Super 4PCS: Fast Global Pointcloud Registration via Smart Indexing, [[paper](https://geometry.cs.ucl.ac.uk/projects/2014/super4PCS/super4pcs.pdf)] [[code](https://github.com/nmellado/Super4PCS)]

***2012:***

**[IJRR]** Rigid 3D Geometry Matching for Grasping of Known Objects in Cluttered Scenes, [[paper](https://mediatum.ub.tum.de/doc/1285820/document.pdf)] [[code](https://github.com/tum-mvp/ObjRecRANSAC)]

***2011:***

**[ICCVW]** CAD-model recognition and 6DOF pose estimation using 3D cues, [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6130296)]

***2010:***

**[CVPR]** Model Globally, Match Locally: Efficient and Robust 3D Object Recognition, [[paper](http://campar.in.tum.de/pub/drost2010CVPR/drost2010CVPR.pdf)] [[code](https://github.com/adrelino/ppf-reconstruction)]

***2009:***

**[ICRA]** Fast Point Feature Histograms (FPFH) for 3D registration, [[paper](https://www.cvl.iis.u-tokyo.ac.jp/class2016/2016w/papers/6.3DdataProcessing/Rusu_FPFH_ICRA2009.pdf)]

***2008:***

**[TOG]** 4-points congruent sets for robust pairwise surface registration, [[paper](http://vecg.cs.ucl.ac.uk/Projects/SmartGeometry/fpcs/paper_docs/fpcs_sig_08.pdf)]

***1981:***

**[ACM]** Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography, [[paper](https://apps.dtic.mil/dtic/tr/fulltext/u2/a460585.pdf)]



#### 1.2 Branch-and-bound techniques

***2020:***

**[arXiv]** DeepGMR: Learning Latent Gaussian Mixture Models for Registration, [[paper](https://arxiv.org/pdf/2008.09088.pdf)]

**[ITSC]** DeepCLR: Correspondence-Less Architecture for Deep End-to-End Point Cloud Registration, [[paper](https://arxiv.org/pdf/2007.11255.pdf)]

**[arXiv]** Aligning Partially Overlapping Point Sets: an Inner Approximation Algorithm, [[paper](https://arxiv.org/pdf/2007.02363.pdf)]

**[arXiv]** Minimum Potential Energy of Point Cloud for Robust Global Registration, [[paper](https://arxiv.org/pdf/2006.06460.pdf)]

**[arXiv]** A Dynamical Perspective on Point Cloud Registration, [[paper](https://arxiv.org/pdf/2005.03190.pdf)]

**[arXiv]** Feature-metric Registration: A Fast Semi-supervised Approach for Robust Point Cloud Registration without Correspondences, [[paper](https://arxiv.org/pdf/2005.01014.pdf)]

**[CVPR]** Deep Global Registration, [[paper](https://arxiv.org/pdf/2004.11540.pdf)]

**[arXiv]** DPDist : Comparing Point Clouds Using Deep Point Cloud Distance, [[paper](https://arxiv.org/pdf/2004.11784.pdf)]

**[arXiv]** Single Shot 6D Object Pose Estimation, [[paper](https://arxiv.org/pdf/2004.12729.pdf)]

**[arXiv]** A Benchmark for Point Clouds Registration Algorithms, [[paper](https://arxiv.org/pdf/2003.12841.pdf)] [[code](https://github.com/iralabdisco/point_clouds_registration_benchmark)]

**[arXiv]** TEASER: Fast and Certifiable Point Cloud Registration, [[paper](https://arxiv.org/pdf/2001.07715.pdf)] [[code](https://github.com/MIT-SPARK/TEASER-plusplus)]

**[arXiv]** Plane Pair Matching for Efficient 3D View Registration, [[paper](https://arxiv.org/pdf/2001.07058.pdf)]

**[arXiv]** Learning multiview 3D point cloud registration, [[paper](https://arxiv.org/pdf/2001.05119.pdf)]

**[arXiv]** Robust, Occlusion-aware Pose Estimation for Objects Grasped by Adaptive Hands, [[paper](https://arxiv.org/pdf/2003.03518.pdf)]

**[arXiv]** Non-iterative One-step Solution for Point Set Registration Problem on Pose Estimation without Correspondence, [[paper](https://arxiv.org/pdf/2003.00457.pdf)]

**[arXiv]** 6D Object Pose Regression via Supervised Learning on Point Clouds, [[paper](https://arxiv.org/pdf/2001.08942.pdf)]

***2019:***

**[arXiv]** One Framework to Register Them All: PointNet Encoding for Point Cloud Alignment, [[paper](https://arxiv.org/abs/1912.05766)]

**[arXiv]** PCRNet: Point Cloud Registration Network using PointNet Encoding, [[paper](https://arxiv.org/abs/1908.07906)] [[code](https://github.com/vinits5/pcrnet)]

**[CVPR]** The Alignment of the Spheres: Globally-Optimal Spherical Mixture Alignment for Camera Pose Estimation, [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Campbell_The_Alignment_of_the_Spheres_Globally-Optimal_Spherical_Mixture_Alignment_for_CVPR_2019_paper.pdf)]

***2016:***

**[CVPR]** GOGMA: Globally-Optimal Gaussian Mixture Alignment, [[paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Campbell_GOGMA_Globally-Optimal_Gaussian_CVPR_2016_paper.pdf)]

**[TPAMI]** Go-ICP: A Globally Optimal Solution to 3D ICP Point-Set Registration, [[paper](https://arxiv.org/pdf/1605.03344.pdf)] [[code](https://github.com/yangjiaolong/Go-ICP)]



#### 2. Local Registration

##### 2.1 Minimizing distances

***2020:***

**[arXiv]** Unsupervised Learning of 3D Point Set Registration, [[paper](https://arxiv.org/pdf/2006.06200.pdf)]

**[arXiv]** Fast and Robust Iterative Closet Point, [[paper](https://arxiv.org/pdf/2007.07627.pdf)]

**[arXiv]** Applying Lie Groups Approaches for Rigid Registration of Point Clouds, [[paper](https://arxiv.org/pdf/2006.13341.pdf)]

**[arXiv]** An Analysis of SVD for Deep Rotation Estimation, [[paper](https://arxiv.org/pdf/2006.14616.pdf)]

***2019:***

**[CVPR]** PointNetLK: Robust & Efficient Point Cloud Registration using PointNet, [[paper](https://arxiv.org/abs/1903.05711)] [[code](https://github.com/hmgoforth/PointNetLK)]

**[NeurIPS]** PRNet: Self-Supervised Learning for Partial-to-Partial Registration, [[paper](https://arxiv.org/abs/1910.12240)]

**[TOG]** A symmetric objective function for ICP, [[paper](https://dl.acm.org/doi/pdf/10.1145/3306346.3323037)]

***2009:***

**[RSS]** Generalized ICP, [[paper](http://www.robots.ox.ac.uk/~avsegal/resources/papers/Generalized_ICP.pdf)]

***2005:***

**[IVC]** Robust Euclidean alignment of 3D point sets: the Trimmed Iterative Closest Point algorithm, [[paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.76.1500&rep=rep1&type=pdf)]

***2004:***

**[Report]** Linear least-squares optimization for point-to-plane icp surface registration, [[paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.116.7292&rep=rep1&type=pdf)]

***2003:***

**[IVC]** Robust registration of 2D and 3D point sets, [[paper](http://luthuli.cs.uiuc.edu/~daf/courses/Opt-2019/Papers/sdarticle.pdf)]

***2001:***

**[3DDIM]** Efficient variants of the ICP algorithm, [[paper](http://webserver2.tecgraf.puc-rio.br/~mgattass/ra/ref/ICP/fasticp_paper.pdf)]

***1992:***

**[TPAMI]** ICP: A method for registration of 3-D shapes, [[paper](http://www-evasion.inrialpes.fr/people/Franck.Hetroy/Teaching/ProjetsImage/2007/Bib/besl_mckay-pami1992.pdf)]

***1991:***

**[ICRA]** Object modeling by registration of multiple range images, [[paper](https://graphics.stanford.edu/~smr/ICP/comparison/chen-medioni-align-rob91.pdf)]



##### 2.2 Probabilistic registration

***2020:***

**[arXiv]** PointGMM: a Neural GMM Network for Point Clouds, [[paper](https://arxiv.org/pdf/2003.13326.pdf)]

***2019:***

**[CVPR]** FilterReg: Robust and Efficient Probabilistic Point-Set Registration using Gaussian Filter and Twist Parameterization, [[paper](https://arxiv.org/pdf/1811.10136.pdf)] [[code](https://sites.google.com/view/filterreg/home)]

***2018:***

**[ECCV]** HGMR: Hierarchical Gaussian Mixtures for Adaptive 3D Registration, [[paper](http://jankautz.com/publications/hGMM_ECCV18.pdf)]

**[CVPR]** Density Adaptive Point Set Registration, [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Lawin_Density_Adaptive_Point_CVPR_2018_paper.pdf)]

**[CVPR]** Fast Monte-Carlo Localization on Aerial Vehicles using Approximate Continuous Belief Representations, [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Dhawale_Fast_Monte-Carlo_Localization_CVPR_2018_paper.pdf)]

**[RAL]** On-Manifold GMM Registration, [[paper](https://static1.squarespace.com/static/5c48e295620b85be0b03f7af/t/5c57be5371c10bb864157a4b/1549254243767/main+-+Wennie+Tabib.pdf)]

***2017:***

**[TPAMI]** Joint Alignment of Multiple Point Sets with Batch and Incremental Expectation-Maximization, [[paper](https://hal.inria.fr/hal-01413414/document)]

***2012:***

**[IJRR]** Fast and accurate scan registration through minimization of the distance between compact 3D NDT representations, [[paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.818.9757&rep=rep1&type=pdf)]

***2011:***

**[TPAMI]** Robust Point Set Registration Using Gaussian Mixture Models, [[paper](https://github.com/bing-jian/gmmreg/blob/master/gmmreg_PAMI_preprint.pdf)] [[code](https://github.com/bing-jian/gmmreg)]

***2009:***

**[D]** The three-dimensional normal-distributions transform: an efficient representation for registration, surface analysis, and loop detection, [[paper](https://www.diva-portal.org/smash/get/diva2:276162/FULLTEXT02)]

***2002:***

**[ECCV]** Multi-scale EM-ICP: A Fast and Robust Approach for Surface Registration, [[paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.18.5106&rep=rep1&type=pdf)]



#### 3. Applications

***2020:***

**[arXiv]** SceneCAD: Predicting Object Alignments and Layouts in RGB-D Scans, [[paper](https://arxiv.org/pdf/2003.12622.pdf)]

***2019:***

**[ICCV]** End-to-End CAD Model Retrieval and 9DoF Alignment in 3D Scans, [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Avetisyan_End-to-End_CAD_Model_Retrieval_and_9DoF_Alignment_in_3D_Scans_ICCV_2019_paper.pdf)]

***2016:***

**[arXiv]** Lessons from the Amazon Picking Challenge, [[paper](https://arxiv.org/abs/1601.05484v2)]

**[arXiv]** Team Delft's Robot Winner of the Amazon Picking Challenge 2016, [[paper](https://arxiv.org/abs/1610.05514)]