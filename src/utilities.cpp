#include "utilities.hpp"


// utility function for annoying fact that hdf5 interface in c++ does not create the necesary parent
// groups for an hdf5 group. This takes a string specifying a group and creates the parent group if 
// necessary. 
std::string create_parent_group(H5::H5File &h5file, std::string group_name)
{
    char *grpname;
    std::vector<std::string> grpnames;
    std::string parentgrpname = "";

    //first create necessary groups
    grpname = std::strtok(group_name.data(), "/");
    while(grpname){
        grpnames.push_back(std::string(grpname));
        grpname = strtok(NULL, "/");
    }
    parentgrpname += grpnames[0];
    if(!h5file.nameExists(parentgrpname))
        h5file.createGroup(parentgrpname);
    
    for(int i=1; i<grpnames.size()-1; i++) //create group for all but last name
    {  
        parentgrpname += "/"+grpnames[i];
        if(!h5file.nameExists(parentgrpname))
            h5file.createGroup(parentgrpname);  
    }
    
    return parentgrpname+"/"+grpnames.back();  
}


void write_slab(H5::DataSet &dataset, Tensor_3D<double> &p, std::vector<hsize_t> &offset, H5::DataSpace &mem_dspace, H5::DataSpace &file_dspace )
{
    int N_particles = (int)(p.dimensions[1]);
    //select memory hyperslab
    std::vector<hsize_t> count = {N_particles,1};
    mem_dspace.selectHyperslab(H5S_SELECT_SET, count.data(), offset.data());
    //select file hyperslab
    std::vector<hsize_t> file_count = {1,N_particles};
    std::vector<hsize_t> file_offset = {offset[1], offset[0]};
    file_dspace.selectHyperslab(H5S_SELECT_SET, file_count.data(), file_offset.data());
    //write data
    dataset.write(p.data(), H5::PredType::NATIVE_DOUBLE, mem_dspace, file_dspace);

}


void write_output(std::string filename, ParticleSet &particles, ParticleSet &tracers, double &current_time, Histogram & hist)
{
    //create file
    H5::H5File h5file(filename, H5F_ACC_TRUNC);
    // get number of particles
    int N_particles = particles.size, N_tracers = tracers.size;
    hsize_t rank;
    std::string datasetname;
    std::vector<hsize_t> offset, filedims;

    //create memory datspace 
    std::vector<hsize_t> datadims = {N_particles,3};
    rank = datadims.size(); //number of dimensions for this dataset
    H5::DataSpace mem_dspace(rank, datadims.data()); //for memory chunk
    datadims = {N_tracers, 3};
    H5::DataSpace tr_mem_dspace(rank, datadims.data());

    //create file data spaces
    filedims = {3,N_particles};
    rank = filedims.size();
    H5::DataSpace file_dspace(rank, filedims.data()); //in file
    filedims = {3, N_tracers};
    rank = filedims.size();
    H5::DataSpace tr_file_dspace(rank, filedims.data());

    //DataSpace to make mu
    filedims = {N_particles};
    H5::DataSpace mu_dspace(1, filedims.data());
    filedims = {N_tracers};
    H5::DataSpace tr_mu_dspace(1, filedims.data());

    //DataSpace to make histogram
    filedims = {hist.counts.size()};
    H5::DataSpace hist_dspace(1, filedims.data());

    //create datasets 
    create_parent_group(h5file, "particles/pos");
    H5::DataSet pos_dataset = h5file.createDataSet("particles/pos", H5::PredType::NATIVE_DOUBLE, file_dspace, H5P_DEFAULT);
    H5::DataSet vel_dataset = h5file.createDataSet("particles/vel", H5::PredType::NATIVE_DOUBLE, file_dspace, H5P_DEFAULT);
    H5::DataSet mu_dataset = h5file.createDataSet("particles/mu", H5::PredType::NATIVE_DOUBLE, mu_dspace, H5P_DEFAULT);

    create_parent_group(h5file, "tracers/pos");
    H5::DataSet tr_pos_dataset = h5file.createDataSet("tracers/pos", H5::PredType::NATIVE_DOUBLE, tr_file_dspace, H5P_DEFAULT);
    H5::DataSet tr_vel_dataset = h5file.createDataSet("tracers/vel", H5::PredType::NATIVE_DOUBLE, tr_file_dspace, H5P_DEFAULT); 
    H5::DataSet tr_mu_dataset = h5file.createDataSet("tracers/mu", H5::PredType::NATIVE_DOUBLE, tr_mu_dspace, H5P_DEFAULT);
    
    H5::DataSet hist_dataset = h5file.createDataSet("group_hist", H5::PredType::NATIVE_DOUBLE, hist_dspace, H5P_DEFAULT);
    

    //write mu's
    mu_dataset.write(particles.mu.data(), H5::PredType::NATIVE_DOUBLE, mu_dspace);
    tr_mu_dataset.write(tracers.mu.data(), H5::PredType::NATIVE_DOUBLE, tr_mu_dspace);

    for(int xyz=0;xyz<3;xyz++) //xyz=0-->, xyz=1-->y, xyz=2 --> z
    {
        offset = {0,xyz}; //pos offset 
        //particles
        write_slab(pos_dataset, particles.pos, offset, mem_dspace, file_dspace);
        write_slab(vel_dataset, particles.vel, offset, mem_dspace, file_dspace);

        //tracers
        write_slab(tr_pos_dataset, tracers.pos, offset, tr_mem_dspace, tr_file_dspace);
        write_slab(tr_vel_dataset, tracers.vel, offset, tr_mem_dspace, tr_file_dspace);
    }
    
    //write the time
    filedims = {1};
    H5::DataSpace time_dspace(1,filedims.data());
    H5::DataSet time_dataset = h5file.createDataSet("time", H5::PredType::NATIVE_DOUBLE, time_dspace, H5P_DEFAULT);
    time_dataset.write(&current_time, H5::PredType::NATIVE_DOUBLE, time_dspace);
    
    //write histogram
    hist_dataset.write(hist.counts.data(), H5::PredType::NATIVE_DOUBLE, hist_dspace);

    h5file.close();
    return;
}

std::vector<hsize_t> get_dataset_dims(H5::H5File &h5file, std::string dsetname)
{
    //open the dataset
    H5::DataSet dataset = h5file.openDataSet(dsetname);
    
    //get the dataspace
    H5::DataSpace dataspace = dataset.getSpace();
    
    //get the rank of the dataset
    int rank = dataspace.getSimpleExtentNdims();
    std::vector<hsize_t> dims(rank);

    //get the size of each dimension
    dataspace.getSimpleExtentDims(dims.data(), NULL);
    return dims;
      
}

void read_slab(H5::DataSet &dataset, Tensor_3D<double> &p, std::vector<hsize_t> &offset, H5::DataSpace &mem_dspace, H5::DataSpace &file_dspace)
{
    hsize_t N_particles = (hsize_t)(p.dimensions[1]);
    //select memory hyperslab
    std::vector<hsize_t> count = {N_particles,1};
    mem_dspace.selectHyperslab(H5S_SELECT_SET, count.data(), offset.data());
    //select file hyperslab
    std::vector<hsize_t> file_count = {1, N_particles};
    std::vector<hsize_t> file_offset = {offset[1], offset[0]};
    file_dspace.selectHyperslab(H5S_SELECT_SET, file_count.data(), file_offset.data());
    //read data
    dataset.read(p.data(), H5::PredType::NATIVE_DOUBLE, mem_dspace, file_dspace); 
}


void file_init(std::string filename, ParticleSet & particles, ParticleSet &tracers, double &current_time)
{
    hsize_t N_tracers, N_particles;
    Tensor_3D<double> pos, vel, trpos, trvel;
    std::vector<double> mu, trmu;
    std::vector<hsize_t> offset;

    //open up the hdf5 file 
    H5::H5File h5file(filename, H5F_ACC_RDONLY);
    
    /***get particles dataset****/
    // get the dimensions of the dataset
    std::vector<hsize_t> file_dims = get_dataset_dims(h5file, "particles/pos");
    H5::DataSpace file_dspace(2,file_dims.data());
    // the memory dimensions will be the transpose:
    N_particles = file_dims[1];
    std::vector<hsize_t> mem_dims = {N_particles, file_dims[0]}; 
    H5::DataSpace mem_dspace(2, mem_dims.data());
    //get the dataset
    H5::DataSet pos_dataset = h5file.openDataSet("particles/pos");
    H5::DataSet vel_dataset = h5file.openDataSet("particles/vel");

    /***get the tracers dataset***/
    file_dims = get_dataset_dims(h5file, "tracers/pos");
    H5::DataSpace tr_file_dspace(2, file_dims.data());
    N_tracers = file_dims[1];
    mem_dims = {N_tracers, file_dims[0]};
    H5::DataSpace tr_mem_dspace(2, mem_dims.data());
    H5::DataSet tr_pos_dataset = h5file.openDataSet("tracers/pos");
    H5::DataSet tr_vel_dataset = h5file.openDataSet("tracers/vel");

    //prepare tensors to be read into
    pos = Tensor_3D<double>(1, N_particles, 3);
    vel = Tensor_3D<double>(1, N_particles, 3);
    mu.resize(N_particles);
    trpos = Tensor_3D<double>(1, N_tracers, 3);
    trvel = Tensor_3D<double>(1, N_tracers, 3);
    trmu.resize(N_tracers);

    for(int xyz=0; xyz<3; xyz++)
    {
        offset = {0,xyz};
        //particles
        read_slab(pos_dataset, pos, offset, mem_dspace, file_dspace);
        read_slab(vel_dataset, vel, offset, mem_dspace, file_dspace);
        
        //tracers
        read_slab(tr_pos_dataset, trpos, offset, tr_mem_dspace, tr_file_dspace);
        read_slab(tr_vel_dataset, trvel, offset, tr_mem_dspace, tr_file_dspace);
    }

    //read the mu's
    //DataSpace to read mu
    file_dims = get_dataset_dims(h5file, "particles/mu");
    H5::DataSpace mu_dspace(1, file_dims.data());
    file_dims = get_dataset_dims(h5file, "tracers/mu");
    H5::DataSpace tr_mu_dspace(1, file_dims.data());
    h5file.openDataSet("particles/mu").read(mu.data(), H5::PredType::NATIVE_DOUBLE, mu_dspace);
    h5file.openDataSet("tracers/mu").read(trmu.data(), H5::PredType::NATIVE_DOUBLE, tr_mu_dspace);

    //read the time
    file_dims = get_dataset_dims(h5file, "time");
    H5::DataSpace time_dspace(1, file_dims.data());
    h5file.openDataSet("time").read(&current_time, H5::PredType::NATIVE_DOUBLE, time_dspace);

    particles = ParticleSet(mu, pos, vel);
    tracers = ParticleSet(trmu, trpos, trvel);

    return;
}
