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

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Visualizzatore PCL"));
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
    pt.setFilterLimits(-1, 1.5);
    pt.filter(*cloud_z);

    //filtro statistical outlier removal
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud_z);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1);
    sor.filter(*filtered_cloud);

    //visualizzazione nuvola dopo filtro SOR
    pcl::visualization::PCLVisualizer::Ptr viewer_filtered(new pcl::visualization::PCLVisualizer("Filtered Cloud"));

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
    vg.setLeafSize(0.02f, 0.02f, 0.02f);
    vg.filter(*clusters_cloud);

    cout<<"PointCloud dopo il filtraggio: " <<clusters_cloud->size() <<" punti.\n";

    //rimozione di piani (pavimento e muro) per evitare errori nel clustering e classificazione
    pcl::PointCloud<pcl::PointXYZ>::Ptr no_floor(new pcl::PointCloud<pcl::PointXYZ>);
    no_floor = clusters_cloud;
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
    pcl::visualization::PCLVisualizer::Ptr viewer_no_floor(new pcl::visualization::PCLVisualizer("Cloud without plane"));

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
    ec.setMinClusterSize(30);
    ec.setMaxClusterSize(2000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(no_floor);
    ec.extract(cluster_vector);

    std::cout<<"Cluster trovati: " <<cluster_vector.size() <<endl;


    //visualizzazione delle nuvole
    while (!viewer_filtered->wasStopped() && !viewer->wasStopped() && !viewer_no_floor->wasStopped()){
        viewer->spinOnce(100);
        viewer_filtered->spinOnce(100);
        viewer_no_floor->spinOnce(100);
    }

    return 0;
}