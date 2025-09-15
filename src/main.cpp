#include <pcl/common/angles.h> 
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/sac_model_cone.h>
#include <pcl/common/common.h>
#include <iostream>
#include <thread>
#include <vector>

using namespace std;

int main() {

    //visualizzazione cones.pcd
    pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ>("../data/cones.pcd", *raw_cloud) == -1) {
        PCL_ERROR("File mancante o formato file sbagliato\n");
        return -1;
    }

    cout <<"Nuvola caricata: " <<raw_cloud->width * raw_cloud->height <<" punti." <<endl;

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Visualizzatore PCL raw"));
    viewer->addPointCloud<pcl::PointXYZ>(raw_cloud, "sample cloud");
    
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> color_handler(raw_cloud, "z");
    viewer->addPointCloud<pcl::PointXYZ>(raw_cloud, color_handler, "cloud_z");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud_z");
    viewer->setCameraPosition(
    -5, 0, 0,     
    0, 0, 0,     
    0, 0, 1      
    );

    //filtro punti asse z
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_z(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PassThrough<pcl::PointXYZ> pt;
    pt.setInputCloud(raw_cloud);
    pt.setFilterFieldName("z");
    pt.setFilterLimits(-1, 1);
    pt.filter(*cloud_z);

    //filtro statistical outlier removal
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud_z);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.5);
    sor.filter(*filtered_cloud);

    //visualizzazione nuvola dopo filtro SOR
    pcl::visualization::PCLVisualizer::Ptr viewer_filtered(new pcl::visualization::PCLVisualizer("cloud dopo filtro z e outlier removal"));

    viewer_filtered->addPointCloud<pcl::PointXYZ>(filtered_cloud, "clean cloud");
    viewer_filtered->setCameraPosition(
    -5, 0, 0,     // posizione camera nello spazio
    0, 0, 0,     // punto che la camera guarda
    0, 0, 1      // l'asse che è sopra (quindi l'asse z)
    );

    //semplificazione della nuvola rimuovendo punti con VoxelGrid
    pcl::PointCloud<pcl::PointXYZ>::Ptr clusters_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(filtered_cloud);
    vg.setLeafSize(0.01f, 0.01f, 0.01f);
    vg.filter(*clusters_cloud);

    cout<<"PointCloud dopo il filtraggio: " <<clusters_cloud->size() <<" punti.\n";

    //rimozione di piani (pavimento e muro) per evitare errori nel clustering e classificazione
    pcl::PointCloud<pcl::PointXYZ>::Ptr no_floor(new pcl::PointCloud<pcl::PointXYZ>);
    no_floor = filtered_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp(new pcl::PointCloud<pcl::PointXYZ>);
    int num_piani = 0;
    
    for(int i=0; i<2; i++){ //toglie muro e pavimento
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.05);
        seg.setInputCloud(no_floor);
        seg.segment(*inliers, *coefficients);

        if(!inliers->indices.empty()){
            cout<<"Piano trovato\n" <<"Punti del piano rimossi:" <<inliers->indices.size() <<endl;
        }

        pcl::ExtractIndices<pcl::PointXYZ> extract;

        extract.setInputCloud(no_floor);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*temp);
        no_floor.swap(temp);

    }
    
    //visualizzazione nuvola dopo rimozione dei piani
    pcl::visualization::PCLVisualizer::Ptr viewer_no_floor(new pcl::visualization::PCLVisualizer("Cloud without planes"));

    viewer_no_floor->addPointCloud<pcl::PointXYZ>(no_floor, "clean cloud");
    viewer_no_floor->setCameraPosition(
    -5, 0, 0,    
    0, 0, 0,     
    0, 0, 1     
    );     

    //divisione dei cluster per preparare la classificazione
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    tree->setInputCloud(no_floor);

    std::vector<pcl::PointIndices> cluster_vector;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.15);
    ec.setMinClusterSize(15);
    ec.setMaxClusterSize(2000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(no_floor);
    ec.extract(cluster_vector);

    std::cout<<"Cluster trovati: " <<cluster_vector.size() <<endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_final_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    //parametri di classificazione coni -----DA AGGIUSTARE -------
    const float min_inlier_ratio = 0.12f;
    const float distance_threshold = 0.07f;
    const float normal_radius = 0.08f;
    //angolo minimo e massimo di apretura per il cono in radianti
    const float min_opening = 5.0f * M_PI/180;
    const float max_opening = 35.0f * M_PI/180;

    //classificazione dei cluster
    for(const auto &indices : cluster_vector){

        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for(int idx : indices.indices){
            cluster->points.push_back(no_floor->points[idx]);
        }
        cluster->width = cluster->points.size();
        cluster->height = 1;
        cluster->is_dense = true;
        
        //controllo se i cluster hanno almeno 10 ounti se no li escludo e li coloro di rosso 
        if (cluster->points.size() < 10) {
            for (const auto &p : cluster->points) {
                pcl::PointXYZRGB q; q.x = p.x; q.y = p.y; q.z = p.z;
                q.r = 255; q.g = 0; q.b = 0;
                colored_final_cloud->points.push_back(q);
            }
            continue;
        }

        //controllo l'atezza del cluster in modo da escludere a priori cluster troppo alti che non possono essere coni
        pcl::PointXYZ min_pt, max_pt;
        pcl::getMinMax3D(*cluster, min_pt, max_pt);

        float height = max_pt.z - min_pt.z;
        if (height > 0.3f || min_pt.z > -0.3f) {
            for (const auto &p : cluster->points) {
                pcl::PointXYZRGB q; q.x = p.x; q.y = p.y; q.z = p.z;
                q.r = 255; q.g = 0; q.b = 0;
                colored_final_cloud->points.push_back(q);
            }
            continue; 
        }

        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        ne.setInputCloud(cluster);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr ntree(new pcl::search::KdTree<pcl::PointXYZ>());
        ne.setSearchMethod(ntree);
        ne.setRadiusSearch(normal_radius);
        ne.compute(*normals);

        pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_CONE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setNormalDistanceWeight(0.05);
        seg.setMaxIterations(20000);
        seg.setDistanceThreshold(distance_threshold);
        seg.setAxis(Eigen::Vector3f(0.0f, 0.0f, 1.0f));
        seg.setEpsAngle(20.0f * M_PI/180.0f);
        seg.setInputCloud(cluster);
        seg.setInputNormals(normals);

        pcl::PointIndices::Ptr inliers_cone(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coeff_cone(new pcl::ModelCoefficients);
        seg.segment(*inliers_cone, *coeff_cone);

        //classificazione cluster con ransac
        bool is_cone = false;
        if (!inliers_cone->indices.empty()){

            float inlier_ratio = static_cast<float>(inliers_cone->indices.size()) / static_cast<float>(cluster->points.size());
            float opening_angle = -1.0f;
            if (coeff_cone->values.size() >= 7) opening_angle = coeff_cone->values[6];

            if (inlier_ratio >= min_inlier_ratio && opening_angle > 0.0f && opening_angle >= min_opening && opening_angle <= max_opening)
            {
                is_cone = true;
            }
        }

        if (is_cone) {
            for (const auto &p : cluster->points) {
                pcl::PointXYZRGB q; q.x = p.x; q.y = p.y; q.z = p.z;
                q.r = 0; q.g = 255; q.b = 0;
                colored_final_cloud->points.push_back(q);
            }
        }else{
            for (const auto &p : cluster->points) {
                pcl::PointXYZRGB q; q.x = p.x; q.y = p.y; q.z = p.z;
                q.r = 255; q.g = 0; q.b = 0;
                colored_final_cloud->points.push_back(q);
            }
        }

    }

    colored_final_cloud->width = colored_final_cloud->points.size();
    colored_final_cloud->height = 1;
    colored_final_cloud->is_dense = true;

    //visualizzazione nuvola finale con verdi i coni e rossi gli ostacoli
    pcl::visualization::PCLVisualizer::Ptr final_viewer(new pcl::visualization::PCLVisualizer("Visualizzatore PCL raw"));
    final_viewer->addPointCloud<pcl::PointXYZRGB>(colored_final_cloud, "sample cloud");
    final_viewer->setCameraPosition(
    -5, 0, 0,     
    0, 0, 0,     
    0, 0, 1      
    );

    //visualizzazione delle nuvole
    while (!viewer_filtered->wasStopped() && !viewer->wasStopped() && !viewer_no_floor->wasStopped() && !final_viewer->wasStopped()){
        viewer->spinOnce(100);
        viewer_filtered->spinOnce(100);
        viewer_no_floor->spinOnce(100);
        final_viewer->spinOnce(100);
    }

    return 0;
}