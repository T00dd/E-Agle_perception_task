#include <pcl/common/angles.h> 
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <iostream>
#include <thread>

using namespace std;

int main() {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ>("../data/cones.pcd", *cloud) == -1) {
        PCL_ERROR("File mancante o formato file sbagliato\n");
        return -1;
    }

    cout <<"Nuvola caricata: " <<cloud->width * cloud->height <<" punti." <<endl;

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Visualizzatore PCL"));
    viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
    
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> color_handler(cloud, "z");
    viewer->addPointCloud<pcl::PointXYZ>(cloud, color_handler, "cloud_z");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud_z");
    viewer->setCameraPosition(
    -5, 0, 0,     // posizione camera nello spazio
    0, 0, 0,     // punto che la camera guarda
    0, 0, 1      // l'asse che è sopra (quindi l'asse z)
    );
    


    /*while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        //std::this_thread::sleep_for(100ms);
    }*/

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_z(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PassThrough<pcl::PointXYZ> pt;
    pt.setInputCloud(cloud);

    //filtro su asse z:
    pt.setFilterFieldName("z");
    pt.setFilterLimits(-1, 1);
    pt.filter(*cloud_z);

    //rimuovere punti isolati:
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud_z);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1);
    sor.filter(*filtered_cloud);

    pcl::visualization::PCLVisualizer::Ptr viewer_filtered(new pcl::visualization::PCLVisualizer("Filtered Cloud"));

    viewer_filtered->addPointCloud<pcl::PointXYZ>(filtered_cloud, "clean cloud");
    viewer_filtered->setCameraPosition(
    -5, 0, 0,     // posizione camera nello spazio
    0, 0, 0,     // punto che la camera guarda
    0, 0, 1      // l'asse che è sopra (quindi l'asse z)
    );

    

    pcl::PointCloud<pcl::PointXYZ>::Ptr clusters_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(filtered_cloud);
    vg.setLeafSize(0.05f, 0.05f, 0.05f);
    vg.filter(*clusters_cloud);

    pcl::visualization::PCLVisualizer::Ptr viewer_clusters(new pcl::visualization::PCLVisualizer("Cluster Cloud"));

    viewer_clusters->addPointCloud<pcl::PointXYZ>(clusters_cloud, "clean cloud");
    viewer_clusters->setCameraPosition(
    -5, 0, 0,     // posizione camera nello spazio
    0, 0, 0,     // punto che la camera guarda
    0, 0, 1      // l'asse che è sopra (quindi l'asse z)
    );

    cout<<"PointCloud dopo il filtraggio: " <<clusters_cloud->width * clusters_cloud->height <<" punti.\n";
    
    while (!viewer_filtered->wasStopped() && !viewer->wasStopped() && !viewer_clusters->wasStopped()){
        viewer->spinOnce(100);
        viewer_filtered->spinOnce(100);
        viewer_clusters->spinOnce(100);
    }

    return 0;
}