#include <pcl/common/angles.h> 
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
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

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);

    //filtro su asse z:
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-1, 1.5);
    pass.filter(*filtered_cloud);

    //rimuovere punti isolati:
    pcl::PointCloud<pcl::PointXYZ>::Ptr clean_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> rem;
    rem.setInputCloud(filtered_cloud);
    rem.setMeanK(50);
    rem.setStddevMulThresh(1.5);
    rem.filter(*clean_cloud);

    pcl::visualization::PCLVisualizer::Ptr viewer_filtered(new pcl::visualization::PCLVisualizer("Filtered Cloud"));

    viewer_filtered->addPointCloud<pcl::PointXYZ>(clean_cloud, "clean cloud");
    viewer_filtered->setCameraPosition(
    -5, 0, 0,     // posizione camera nello spazio
    0, 0, 0,     // punto che la camera guarda
    0, 0, 1      // l'asse che è sopra (quindi l'asse z)
    );

    while (!viewer_filtered->wasStopped() && !viewer->wasStopped()) {
        viewer->spinOnce(100);
        viewer_filtered->spinOnce(100);
    }



    return 0;
}