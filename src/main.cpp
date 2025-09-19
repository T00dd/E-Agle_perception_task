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
    #include <pcl/registration/icp.h>
    #include <iostream>
    #include <thread>
    #include <vector>

    using namespace std;

    pcl::PointCloud<pcl::PointXYZ>::Ptr filter_z_axes(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_unfiltered, int min_z, int max_z);

    pcl::PointCloud<pcl::PointXYZ>::Ptr filter_planes(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_unfiltered, int num_planes, float distant_threshold);

    pcl::PointCloud<pcl::PointXYZ>::Ptr filter_outlier_removal(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_unfiltered, int meanK, int stddevMulThresh);

    void cluster_extract(vector<pcl::PointIndices> &cluster_vector ,const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, float cluster_tolerance, int min_size, int max_size);

    pcl::PointCloud<pcl::PointXYZ>::Ptr makeConeModel(float height, float radius, int slices);

    bool isConeICP(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cluster, double max_corresp, double score_threshold, double base_radius);

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
        cloud_z = filter_z_axes(raw_cloud, -1, 1);

        //semplificazione della nuvola rimuovendo punti con VoxelGrid
        // pcl::PointCloud<pcl::PointXYZ>::Ptr clusters_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        
        // pcl::VoxelGrid<pcl::PointXYZ> vg;
        // vg.setInputCloud(cloud_z);
        // vg.setLeafSize(0.05f, 0.05f, 0.05f);
        // vg.filter(*clusters_cloud);

        // cout<<"PointCloud dopo il filtraggio: " <<clusters_cloud->size() <<" punti.\n";

        //rimozione di piani (pavimento e muro) per evitare errori nel clustering e classificazione
        pcl::PointCloud<pcl::PointXYZ>::Ptr no_planes(new pcl::PointCloud<pcl::PointXYZ>);
        no_planes = filter_planes(cloud_z, 2, 0.05);

        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        filtered_cloud = filter_outlier_removal(no_planes, 50, 0.2);
        
        
        //visualizzazione nuvola dopo rimozione dei piani
        pcl::visualization::PCLVisualizer::Ptr viewer_no_floor(new pcl::visualization::PCLVisualizer("Cloud without planes"));

        viewer_no_floor->addPointCloud<pcl::PointXYZ>(no_planes, "clean cloud");
        viewer_no_floor->setCameraPosition(
        -5, 0, 0,    
        0, 0, 0,     
        0, 0, 1     
        );     
        
        //divisione dei cluster per preparare la classificazione
        vector<pcl::PointIndices> cluster_vector;
        cluster_extract(cluster_vector, no_planes, 0.15, 15, 2000);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_final_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        //classificazione dei cluster
        int cluster_id = 0;

        //vettore per salvare i centroidi dei coni rilevati
        vector<Eigen::Vector3f> cone_centers;

        for(const auto &indices : cluster_vector){

            //prendo i cluster e li metto in una nuvola
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
            cluster->reserve(indices.indices.size());
            for (int idx : indices.indices)
                cluster->points.push_back(no_planes->points[idx]);

            cluster->width = cluster->points.size();
            cluster->height = 1;
            cluster->is_dense = true;

            pcl::PointCloud<pcl::PointXYZ> cluster_clean;
            std::vector<int> valid_indices;
            pcl::removeNaNFromPointCloud(*cluster, cluster_clean, valid_indices);
            *cluster = cluster_clean;

            cout <<"Cluster #" <<cluster_id++ << "-> punti: " <<cluster->size() <<endl;

            //scarto cluster troppo piccoli
            if (cluster->size() < 20){
                for (const auto &p : cluster->points){
                    pcl::PointXYZRGB q{p.x, p.y, p.z, 255, 0, 0};
                    colored_final_cloud->points.push_back(q);
                }
                cout<<"Cluster #" <<cluster_id <<" scartato: pochi punti\n";
                continue;
            }
           

            //calcolo bounding box
            pcl::PointXYZ min_pt, max_pt;
            pcl::getMinMax3D(*cluster, min_pt, max_pt);

            float height = max_pt.z - min_pt.z;
            float base_radius = std::max(max_pt.x - min_pt.x, max_pt.y - min_pt.y) * 0.5f;

            //controllo altezza e posizione del cluster (scarto quelli irrealistici)
            if (height <= 0.0f || base_radius <= 0.0f || height > 0.3f || min_pt.z > -0.3f){
                for (const auto &p : cluster->points){
                    pcl::PointXYZRGB q{p.x, p.y, p.z, 255, 0, 0};
                    colored_final_cloud->points.push_back(q);
                }
                cout<<"Cluster #" <<cluster_id <<" scartato: posizione o altezza irrealistici\n";
                continue;
            }

            float maxCorrDist = 0.03;
            double score_threshold = 0.01;

            bool is_cone = isConeICP(cluster, maxCorrDist, score_threshold, base_radius); 

            if (is_cone) {
                //coloro il cluster di azzurro se è un cono
                for (const auto &p : cluster->points) {
                    pcl::PointXYZRGB q; q.x = p.x; q.y = p.y; q.z = p.z;
                    q.r = 0; q.g = 200; q.b = 255;
                    colored_final_cloud->points.push_back(q);
                }

                //calcolo e salvo il centroide del cluster per il tracciato
                Eigen::Vector4f c; 
                pcl::compute3DCentroid(*cluster, c);
                cone_centers.emplace_back(c[0], c[1], c[2]);

            }else{
                for (const auto &p : cluster->points) {
                    //coloro di rosso gli ostacoli
                    pcl::PointXYZRGB q; q.x = p.x; q.y = p.y; q.z = p.z;
                    q.r = 255; q.g = 0; q.b = 0;
                    colored_final_cloud->points.push_back(q);
                }
                cout<<"Cluster #" <<cluster_id <<" scartato: non è un cono\n";
            }
        }

        colored_final_cloud->width = colored_final_cloud->points.size();
        colored_final_cloud->height = 1;
        colored_final_cloud->is_dense = true;

        //visualizzazione nuvola finale con verdi i coni e rossi gli ostacoli
        pcl::visualization::PCLVisualizer::Ptr cone_viewer(new pcl::visualization::PCLVisualizer("Visualizzatore PCL raw"));
        cone_viewer->addPointCloud<pcl::PointXYZRGB>(colored_final_cloud, "sample cloud");
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
        vector<Eigen::Vector3f> right_cones_unordered;
        vector<Eigen::Vector3f> left_cones_unordered;

        for(const auto &c : cone_centers){
            if(c.y() <= 0){
                right_cones_unordered.push_back(c);
            }else{
                left_cones_unordered.push_back(c);
            }
        }

        //funzione NN per ordinare un vettore di punti
        auto order_by_nn = [](const std::vector<Eigen::Vector3f> &pts) {
            std::vector<Eigen::Vector3f> ordered;
            if (pts.empty()) return ordered;

            //trova indice di partenza: min distanza dall'origine
            int start_idx = 0;
            float best_d = std::numeric_limits<float>::max();
            for (size_t i=0;i<pts.size();++i){
                float d = pts[i].head<2>().squaredNorm();
                if (d < best_d){ best_d=d; start_idx=(int)i; }
            }

            std::vector<char> used(pts.size(),0);
            int current = start_idx;
            ordered.push_back(pts[current]);
            used[current] = 1;

            for (size_t step = 1; step < pts.size(); ++step) {
                float best = std::numeric_limits<float>::max();
                int idx = -1;
                for (size_t j = 0; j < pts.size(); ++j) {
                    if (used[j]) continue;
                    float d = (pts[j] - pts[current]).head<2>().squaredNorm(); //distanza XY
                    if (d < best){ best = d; idx = j; }
                }
                if (idx == -1) break;
                current = idx;
                used[current] = 1;
                ordered.push_back(pts[current]);
            }
            return ordered;
        };

        auto left_cones = order_by_nn(left_cones_unordered);
        auto right_cones = order_by_nn(right_cones_unordered);

        // if (!cone_centers.empty())
        // {
        //     // trova l'indice di partenza: cono più vicino all'origine (0,0,0)
        //     int start_idx = 0;
        //     float best_d = std::numeric_limits<float>::max();
        //     for (size_t i = 0; i < cone_centers.size(); ++i) {
        //         float d = cone_centers[i].norm();
        //         if (d < best_d) { best_d = d; start_idx = (int)i; }
                
        //         if(cone_centers[i].y() <= 0){
        //             right_cones.reserve(cone_centers.size());
        //             std::vector<char> used(cone_centers.size(), 0);
        //             int current = start_idx;
        //             right_cones.push_back(cone_centers[current]);
        //             used[current] = 1;
        //         }else{
        //             left_cones.reserve(cone_centers.size());
        //             std::vector<char> used(cone_centers.size(), 0);
        //             int current = start_idx;
        //             left_cones.push_back(cone_centers[current]);
        //             used[current] = 1;
        //         }
        //     }   

        //     for (size_t step = 1; step < cone_centers.size(); ++step) {
        //         float best_dist = std::numeric_limits<float>::max();
        //         int best_idx = -1;
        //         for (size_t j = 0; j < cone_centers.size(); ++j) {
        //             if (used[j]) continue;
        //             float d = (cone_centers[j] - cone_centers[current]).squaredNorm();
        //             if (d < best_dist) { best_dist = d; best_idx = (int)j; }
        //         }
        //         if (best_idx == -1) break;
        //         current = best_idx;
        //         used[current] = 1;
        //         path.push_back(cone_centers[current]);
        //     }
        // }

        //visualizzatore percorso
        pcl::visualization::PCLVisualizer::Ptr track_viewer(new pcl::visualization::PCLVisualizer("Track Viewer"));
        track_viewer->setBackgroundColor(0, 0, 0);
        track_viewer->addCoordinateSystem(1.0);   

        //bordo destra del tracciato -> VERDE
        for (size_t i = 0; i < right_cones.size(); ++i){
            pcl::PointXYZ center(right_cones[i].x(), right_cones[i].y(), right_cones[i].z());
            string sph_id = "track_right_sphere_" + to_string(i);
            track_viewer->addSphere(center, 0.03, 1, 1, 0, sph_id);

            if (i > 0) {
                pcl::PointXYZ a(right_cones[i-1].x(), right_cones[i-1].y(), right_cones[i-1].z());
                pcl::PointXYZ b(right_cones[i].x(),   right_cones[i].y(),   right_cones[i].z());
                string line_id = "track_right_line_" + to_string(i);
                track_viewer->addLine(a, b, 0, 1, 0, line_id); // verde
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
                track_viewer->addLine(a, b, 0, 1, 0, line_id); // verde
            }
        }

        //visualizzazione delle nuvole
        while (!track_viewer->wasStopped() && !viewer->wasStopped() && !viewer_no_floor->wasStopped() && !cone_viewer->wasStopped()){
            viewer->spinOnce(100);
            track_viewer->spinOnce(100);
            viewer_no_floor->spinOnce(100);
            cone_viewer->spinOnce(100);
        }

        return 0;
    }


    pcl::PointCloud<pcl::PointXYZ>::Ptr filter_z_axes(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_unfiltered, int min_z, int max_z){ 

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

        for(int i=0; i<num_planes; i++){ //toglie muro e pavimento
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

    void cluster_extract(vector<pcl::PointIndices> &cluster_vector ,const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, float cluster_tolerance, int min_size, int max_size){ 

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

    pcl::PointCloud<pcl::PointXYZ>::Ptr makeConeModel(float height, float radius, int slices){

        float fov_deg = 200.0f;          //angolo visibile dal LiDAR
        float slope_eps = 0.02f;         //piccola pendenza per evitare superfici piatte
        auto model = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        model->points.reserve(slices * 50);

        const float fov_rad = fov_deg * M_PI / 180.0f;
        const float start_angle = -fov_rad / 2.0f;
        const float end_angle   =  fov_rad / 2.0f;
        const int vertical_steps = 50;

        for (int i = 0; i < slices; ++i)
        {
            float angle = start_angle + (end_angle - start_angle) * i / (slices - 1);
            

            for (int k = 0; k <= vertical_steps; ++k)
            {
                float z = height * k / vertical_steps;
                float r = radius * (1.0f - z / height);

                pcl::PointXYZ p;
                p.x = r * cos(angle);
                p.y = r * sin(angle);

                //sposta leggermente lungo Y in base a Z,
                //così non esistono colonne perfettamente verticali
                p.y += slope_eps * z;

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

        double score = icp.getFitnessScore(); //media distanze^2
        cout <<"ICP fitness score: " <<score <<endl;

        return score < score_threshold;
    }