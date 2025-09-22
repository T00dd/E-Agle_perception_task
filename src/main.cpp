    #include <pcl/io/pcd_io.h>
    #include <pcl/visualization/pcl_visualizer.h>
    #include <pcl/console/parse.h>
    #include <pcl/filters/passthrough.h>
    #include <pcl/filters/statistical_outlier_removal.h>
    #include <pcl/filters/voxel_grid.h>
    #include <pcl/search/kdtree.h>
    #include <pcl/segmentation/extract_clusters.h>
    #include <pcl/segmentation/sac_segmentation.h>
    #include <pcl/filters/extract_indices.h>
    #include <pcl/sample_consensus/sac_model_cone.h>
    #include <pcl/common/common.h>
    #include <pcl/registration/icp.h>
    #include <iostream>
    #include <omp.h>
    #include <vector>

    using namespace std;

    pcl::PointCloud<pcl::PointXYZ>::Ptr filter_z_axes(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_unfiltered, float min_z, float max_z);

    pcl::PointCloud<pcl::PointXYZ>::Ptr filter_planes(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_unfiltered, int num_planes, float distant_threshold);

    pcl::PointCloud<pcl::PointXYZ>::Ptr filter_outlier_removal(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_unfiltered, int meanK, int stddevMulThresh);

    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampling_voxelgrid(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_unfiltered, float leaf_size);

    void cluster_extraction(vector<pcl::PointIndices> &cluster_vector ,const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, float cluster_tolerance, int min_size, int max_size);

    pcl::PointCloud<pcl::PointXYZ>::Ptr makeConeModel(float height, float radius, int slices);

    bool isConeICP(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cluster, double max_corresp, double score_threshold, double base_radius);

    void odometry(const pcl::PointCloud<pcl::PointXYZ>::Ptr &first, const pcl::PointCloud<pcl::PointXYZ>::Ptr &second, int max_iteration, float max_correspond_distance);

    void order_by_nn(std::vector<Eigen::Vector3f> &pts); 

    int main() {
        
        cout <<"Numero di thread disponibili: " <<omp_get_max_threads() <<endl;

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
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_z");
        viewer->setCameraPosition(
        -5, 0, 0,     
        0, 0, 0,     
        0, 0, 1      
        );

        //FILTRO ASSE Z
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_z(new pcl::PointCloud<pcl::PointXYZ>);
        cloud_z = filter_z_axes(raw_cloud, -1, 1);

        //RIMOZIONE PIANI (pavimento e muro) per evitare errori nel clustering e classificazione
        pcl::PointCloud<pcl::PointXYZ>::Ptr no_planes(new pcl::PointCloud<pcl::PointXYZ>);
        no_planes = filter_planes(cloud_z, 2, 0.05);

        //FILTRO OUTLIER REMOVAL
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        filtered_cloud = filter_outlier_removal(no_planes, 20, 1);
        
        //visualizzazione nuvola dopo pipline
        pcl::visualization::PCLVisualizer::Ptr viewer_no_floor(new pcl::visualization::PCLVisualizer("Cloud without planes"));
        viewer_no_floor->addPointCloud<pcl::PointXYZ>(filtered_cloud, "clean cloud");
        viewer_no_floor->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.5, "clean cloud");
        viewer_no_floor->setCameraPosition(
        -5, 0, 0,    
        0, 0, 0,     
        0, 0, 1     
        );
        
        //DIVISIONE CLUSTER
        vector<pcl::PointIndices> cluster_vector;
        cluster_extraction(cluster_vector, filtered_cloud, 0.15, 15, 2000);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_final_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        //CLASSIFICAZIONE CLUSTER

        //vettore per salvare i centroidi dei coni rilevati
        vector<Eigen::Vector3f> cone_centers;

        #pragma omp parallel for
        for(int i = 0; i<cluster_vector.size(); i++){

            //prendo i cluster e li metto in una nuvola
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
            cluster->reserve(cluster_vector[i].indices.size());
            for (int idx : cluster_vector[i].indices)
                cluster->points.push_back(filtered_cloud->points[idx]);

            cluster->width = cluster->points.size();
            cluster->height = 1;
            cluster->is_dense = true;

            pcl::PointCloud<pcl::PointXYZ> cluster_clean;
            std::vector<int> valid_indices;
            pcl::removeNaNFromPointCloud(*cluster, cluster_clean, valid_indices);
            *cluster = cluster_clean;

            #pragma omp critical
            {
                cout <<"Cluster #" <<i <<" -> punti: " <<cluster->size() <<endl;
            }

            //scarto cluster troppo piccoli
            if (cluster->size() < 20){
                #pragma omp critical
                {
                    for (const auto &p : cluster->points){
                    pcl::PointXYZRGB q{p.x, p.y, p.z, 255, 0, 0};
                    colored_final_cloud->points.push_back(q);
                    }
                    cout<<"Cluster #" <<i <<" scartato: pochi punti\n";
                }
                continue;
            }
           

            //calcolo bounding box
            pcl::PointXYZ min_pt, max_pt;
            pcl::getMinMax3D(*cluster, min_pt, max_pt);

            float height = max_pt.z - min_pt.z;
            float base_radius = std::max(max_pt.x - min_pt.x, max_pt.y - min_pt.y) * 0.5f;

            //controllo altezza e posizione del cluster (scarto quelli irrealistici)
            if (height <= 0.0f || base_radius <= 0.0f || height > 0.3f || min_pt.z > -0.3f){
                #pragma omp critical
                {
                    for (const auto &p : cluster->points){
                        pcl::PointXYZRGB q{p.x, p.y, p.z, 255, 0, 0};
                        colored_final_cloud->points.push_back(q);
                    }
                    cout<<"Cluster #" <<i <<" scartato: posizione o altezza irrealistici\n";                    
                }
                continue;
            }

            float maxCorrDist = 0.03;
            double score_threshold = 0.01;

            bool is_cone = isConeICP(cluster, maxCorrDist, score_threshold, base_radius); 

            if (is_cone) {
                Eigen::Vector4f c; 
                pcl::compute3DCentroid(*cluster, c);

                //coloro il cluster di azzurro se è un cono
                #pragma omp critical
                {
                    for (const auto &p : cluster->points) {
                        pcl::PointXYZRGB q; q.x = p.x; q.y = p.y; q.z = p.z;
                        q.r = 0; q.g = 200; q.b = 255;
                        colored_final_cloud->points.push_back(q);
                    }
                    cone_centers.emplace_back(c[0], c[1], c[2]);
                }
                

            }else{

                #pragma omp critical
                {
                    for (const auto &p : cluster->points) {
                        pcl::PointXYZRGB q; q.x = p.x; q.y = p.y; q.z = p.z;
                        q.r = 255; q.g = 0; q.b = 0;
                        colored_final_cloud->points.push_back(q);
                    }
                    cout<<"Cluster #" <<i <<" scartato: non è un cono\n";
                }
            }
        }

        colored_final_cloud->width = colored_final_cloud->points.size();
        colored_final_cloud->height = 1;
        colored_final_cloud->is_dense = true;

        //visualizzazione nuvola finale con verdi i coni e rossi gli ostacoli
        pcl::visualization::PCLVisualizer::Ptr cone_viewer(new pcl::visualization::PCLVisualizer("Visualizzatore PCL raw"));
        cone_viewer->addPointCloud<pcl::PointXYZRGB>(colored_final_cloud, "sample cloud");
        cone_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.5, "sample cloud");
        cone_viewer->setCameraPosition(
        -5, 0, 0,     
        0, 0, 0,     
        0, 0, 1      
        );

        std::cout <<"DEBUG: total cone_centers = " << cone_centers.size() << std::endl;
        for (size_t i = 0; i < cone_centers.size(); ++i) {
            std::cout << "  cone[" << i << "] = (" 
                    << cone_centers[i].x() << ", "
                    << cone_centers[i].y() << ", "
                    << cone_centers[i].z() << ")\n";
        }

        //costruzione percorso (nearest-neighbor ordering)
        vector<Eigen::Vector3f> right_cones;
        vector<Eigen::Vector3f> left_cones;

        for(const auto &c : cone_centers){
            if(c.y() <= 0){
                right_cones.push_back(c);
            }else{
                left_cones.push_back(c);
            }
        }

        order_by_nn(left_cones);
        order_by_nn(right_cones);

        //visualizzatore percorso
        pcl::visualization::PCLVisualizer::Ptr track_viewer(new pcl::visualization::PCLVisualizer("Track Viewer"));
        track_viewer->setBackgroundColor(0, 0, 0);
        track_viewer->addCoordinateSystem(1.0);   
        track_viewer->setCameraPosition(
        -5, 0, 0,     
        0, 0, 0,     
        0, 0, 1      
        );

        //bordo destra del tracciato -> VERDE
        for (size_t i = 0; i < right_cones.size(); ++i){
            pcl::PointXYZ center(right_cones[i].x(), right_cones[i].y(), right_cones[i].z());
            string sph_id = "track_right_sphere_" + to_string(i);
            track_viewer->addSphere(center, 0.03, 1, 1, 0, sph_id);

            if (i > 0) {
                pcl::PointXYZ a(right_cones[i-1].x(), right_cones[i-1].y(), right_cones[i-1].z());
                pcl::PointXYZ b(right_cones[i].x(),   right_cones[i].y(),   right_cones[i].z());
                string line_id = "track_right_line_" + to_string(i);
                track_viewer->addLine(a, b, 0, 1, 0, line_id);
            }
        }

        //bordo sinistra del tracciato -> AZZURRO
        for (size_t i = 0; i < left_cones.size(); ++i){
            pcl::PointXYZ center(left_cones[i].x(), left_cones[i].y(), left_cones[i].z());
            string sph_id = "track_left_sphere_" + to_string(i);
            track_viewer->addSphere(center, 0.03, 1, 1, 0, sph_id);

            if (i > 0) {
                pcl::PointXYZ a(left_cones[i-1].x(), left_cones[i-1].y(), left_cones[i-1].z());
                pcl::PointXYZ b(left_cones[i].x(),   left_cones[i].y(),   left_cones[i].z());
                string line_id = "track_left_line_" + to_string(i);
                track_viewer->addLine(a, b, 0, 1, 0, line_id);
            }
        }

        //ODOMETRY

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZ>);

        if (pcl::io::loadPCDFile<pcl::PointXYZ>("../data/first.pcd", *cloud1) == -1 || pcl::io::loadPCDFile<pcl::PointXYZ>("../data/second.pcd", *cloud2) == -1){
            PCL_ERROR("File mancante o formato file sbagliato\n");
            return -1;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1_filter_z(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2_filter_z(new pcl::PointCloud<pcl::PointXYZ>);

        cloud1_filter_z = filter_z_axes(cloud1, -1, 1);
        cloud2_filter_z = filter_z_axes(cloud2, -1, 1);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1_no_planes(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2_no_planes(new pcl::PointCloud<pcl::PointXYZ>);

        cloud1_no_planes = filter_planes(cloud1_filter_z, 2, 0.05);
        cloud2_no_planes = filter_planes(cloud2_filter_z, 2, 0.05);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1_outlier_rem(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2_outlier_rem(new pcl::PointCloud<pcl::PointXYZ>);

        cloud1_outlier_rem = filter_outlier_removal(cloud1_no_planes, 80, 0.2);
        cloud2_outlier_rem = filter_outlier_removal(cloud2_no_planes, 80, 0.2);
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2_downsampled(new pcl::PointCloud<pcl::PointXYZ>);

        cloud1_downsampled = downsampling_voxelgrid(cloud1_outlier_rem, 0.05f);
        cloud2_downsampled = downsampling_voxelgrid(cloud2_outlier_rem, 0.05f);

        odometry(cloud1_downsampled, cloud2_downsampled, 50, 1);

        //visualizzazione delle nuvole
        while (!track_viewer->wasStopped() && !viewer->wasStopped() && !viewer_no_floor->wasStopped() && !cone_viewer->wasStopped()){
            viewer->spinOnce(100);
            track_viewer->spinOnce(100);
            viewer_no_floor->spinOnce(100);
            cone_viewer->spinOnce(100);
        }

        return 0;
    }


    pcl::PointCloud<pcl::PointXYZ>::Ptr filter_z_axes(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_unfiltered, float min_z, float max_z){ 

        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);

        pcl::PassThrough<pcl::PointXYZ> pt;
        pt.setInputCloud(cloud_unfiltered);
        pt.setFilterFieldName("z");
        pt.setFilterLimits(min_z, max_z);
        pt.filter(*filtered);

        return filtered;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr filter_planes(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_unfiltered, int num_planes, float distant_threshold){ 

        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
        filtered = cloud_unfiltered;
        pcl::PointCloud<pcl::PointXYZ>::Ptr temp(new pcl::PointCloud<pcl::PointXYZ>);

        for(int i=0; i<num_planes; i++){
            pcl::SACSegmentation<pcl::PointXYZ> seg;
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_PLANE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(distant_threshold);
            seg.setInputCloud(filtered);
            seg.segment(*inliers, *coefficients);

            if(!inliers->indices.empty()){
                cout<<"Piano trovato\n" <<"Punti del piano rimossi:" <<inliers->indices.size() <<endl;
            }

            pcl::ExtractIndices<pcl::PointXYZ> extract;

            extract.setInputCloud(filtered);
            extract.setIndices(inliers);
            extract.setNegative(true);
            extract.filter(*temp);
            filtered.swap(temp);

        }

        return filtered;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr filter_outlier_removal(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_unfiltered, int meanK, int stddevMulThresh){ 

        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);

        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud_unfiltered);
        sor.setMeanK(meanK);
        sor.setStddevMulThresh(stddevMulThresh);
        sor.filter(*filtered);

        return filtered;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampling_voxelgrid(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_unfiltered, float leaf_size){ 

        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
        
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(cloud_unfiltered);
        vg.setLeafSize(leaf_size, leaf_size, leaf_size);
        vg.filter(*filtered);

        cout<<"PointCloud dopo il filtraggio: " <<filtered->size() <<" punti.\n";

        return filtered;
    }

    void cluster_extraction(vector<pcl::PointIndices> &cluster_vector ,const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, float cluster_tolerance, int min_size, int max_size){ 

        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        tree->setInputCloud(cloud);

        
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(cluster_tolerance);
        ec.setMinClusterSize(min_size);
        ec.setMaxClusterSize(max_size);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_vector);

        cout<<"Cluster trovati: " <<cluster_vector.size() <<endl;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr makeConeModel(float height, float radius, int slices) {

    float fov_deg = 180.0f; //angolo visibile dal LiDAR
    auto model = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    model->points.reserve(slices * 50);

    const float fov_rad = fov_deg * M_PI / 180.0f;
    const float start_angle = -fov_rad / 2.0f;
    const float end_angle   =  fov_rad / 2.0f;
    const int vertical_steps = 50;

    for (int i = 0; i < slices; ++i) {
        float angle = start_angle + (end_angle - start_angle) * i / (slices - 1);

        for (int k = 0; k <= vertical_steps; ++k) {
            float z = height * k / vertical_steps;
            float r = radius * (1.0f - z / height);

            pcl::PointXYZ p;
            p.x = r * cos(angle);
            p.y = r * sin(angle); 
            p.z = z;

            model->points.push_back(p);
        }
    }

    model->width = static_cast<uint32_t>(model->points.size());
    model->height = 1;
    model->is_dense = true;
    return model;
    }


    bool isConeICP(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cluster, double max_corresp, double score_threshold, double base_radius){
        if (cluster->empty()) return false;
        
        //stima bounding box del cluster
        pcl::PointXYZ min_pt, max_pt;
        pcl::getMinMax3D(*cluster, min_pt, max_pt);
        float height = 0.28f;

        //genera modello teorico e lo centra al cluster
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cluster, centroid);
        auto cone_model = makeConeModel(height, base_radius, 50);
        for (auto &p : cone_model->points) {
            p.x += centroid[0];
            p.y += centroid[1];
            p.z += centroid[2];
        }

        //icp
        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        icp.setInputSource(cone_model);
        icp.setInputTarget(cluster);
        icp.setMaximumIterations(20);
        icp.setMaxCorrespondenceDistance(max_corresp);
        icp.setTransformationEpsilon(1e-6);

        pcl::PointCloud<pcl::PointXYZ> aligned;
        icp.align(aligned);

        if (!icp.hasConverged())
            return false;

        double score = icp.getFitnessScore();
        cout <<"ICP fitness score: " <<score <<endl;

        return score < score_threshold;
    }

    void odometry(const pcl::PointCloud<pcl::PointXYZ>::Ptr &first, const pcl::PointCloud<pcl::PointXYZ>::Ptr &second, int max_iteration, float max_correspond_distance){

        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        icp.setMaximumIterations(max_iteration);
        icp.setMaxCorrespondenceDistance(max_correspond_distance);
        icp.setTransformationEpsilon(1e-8);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setInputSource(first);
        icp.setInputTarget(second);

        pcl::PointCloud<pcl::PointXYZ> aligned;
        icp.align(aligned);

        if(icp.hasConverged()){
            cout<<"ICP converged! Fitness score: " <<icp.getFitnessScore() <<endl;
            Eigen::Matrix4f transform = icp.getFinalTransformation();
            cout <<"Trasformazione stimata (odometria):\n" <<transform <<endl;
        }else{
            cout<<"ICP has not converged\n";
        }

    }

    void order_by_nn(std::vector<Eigen::Vector3f> &pts) {
            
        std::vector<Eigen::Vector3f> ordered;
        if (pts.empty()) return;

        int start_idx = 0;
        float best_d = numeric_limits<float>::max();

        for (size_t i=0;i<pts.size();++i){
            float d = pts[i].head<2>().squaredNorm();
            if (d < best_d){ best_d=d; start_idx=(int)i; }
        }

        vector<char> used(pts.size(),0);
        int current = start_idx;
        ordered.push_back(pts[current]);
        used[current] = 1;

        for (size_t step = 1; step < pts.size(); ++step) {
            float best = numeric_limits<float>::max();
            int idx = -1;
            
            for (size_t j = 0; j < pts.size(); ++j) {
                if (used[j]) continue;
                float d = (pts[j] - pts[current]).head<2>().squaredNorm();
                if (d < best){ best = d; idx = j; }
            }
            
            if (idx == -1) break;
            
            current = idx;
            used[current] = 1;
            ordered.push_back(pts[current]);
        }

        pts.swap(ordered);
    }