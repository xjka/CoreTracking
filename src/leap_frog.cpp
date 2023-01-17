#include <cmath>
#include <iostream>
#include <H5Cpp.h>
#include <boost/program_options.hpp> 
#include <random>
#include <vector>
#include <sstream>
#include <eigen3/Eigen/Dense>
#include <list>
#include <utility>
#include <mutex>
#include <thread>
#include <condition_variable>


template <class class_t> class Tensor_3D //as of now row major order
{
    public:
        std::vector<class_t> idata;
        std::array<size_t,3> strides;
        std::array<size_t,3> dimensions;
        
        Tensor_3D(void) : idata(NULL), dimensions{0,0,0}, strides{0,0,0} {}
            
        Tensor_3D(size_t dim1, size_t dim2, size_t dim3) : dimensions{dim1, dim2, dim3}
        {
            strides = std::array<size_t,3>{dim2*dim3, dim3, 1};
            idata = std::vector<class_t>(dim1*dim2*dim3);
        }
        
        class_t &operator()(int idx2, int idx3)
        {
            return idata.at(0*strides[0] + idx2*strides[1] + idx3*strides[2]);
        }

        class_t &operator()(int idx1, int idx2, int idx3)
        {
            return idata.at(idx1*strides[0] + idx2*strides[1] + idx3*strides[2]);
        }
        
        void append(std::vector<class_t> element)
        {
            //technically should check that len(element)=dimensions[2]
            dimensions[2] = element.size();
            for (int i=0; i<element.size(); i++)
            {
                idata.push_back(element[i]);
            }
            dimensions[1] += 1;
            //strides[0] += dimensions[2];
        }

        double* data()
        {
            return idata.data();
        }
};

class ParticleSet {
    
    public:
        std::vector<double> mu;
        long size;
        Tensor_3D <double> pos;
        Tensor_3D <double> vel;
        
        ParticleSet(void) : size(0) {}

        ParticleSet(long size, double mu) : size(size)
        {
            ParticleSet::mu = std::vector<double>(size, mu);
            ParticleSet::pos = Tensor_3D<double>(1,size,3);
            ParticleSet::vel = Tensor_3D<double>(1,size,3);
            ParticleSet::size = size;
        }

        ParticleSet(std::vector<double> mu, Tensor_3D<double> pos, Tensor_3D<double> vel)
        {
            ParticleSet::mu = mu;
            ParticleSet::pos = pos;
            ParticleSet::vel = vel;
            ParticleSet::size = pos.dimensions[1];
        }
};

typedef Eigen::Map<Eigen::ArrayXd> ArrayRef;
typedef Eigen::ArrayXd Array;
class CalcArgs
{
    public:
        int block_size;
        double mu;
        double epsilon;
        double r_tst;
        double v_rms;
        double v_mean;
        int start;
        int stop;
        ArrayRef ax;
        ArrayRef ay;
        ArrayRef az;
        ParticleSet *particles;
        ParticleSet *tracers;
    
    CalcArgs(void) : ax(NULL,0), ay(NULL,0), az(NULL,0) {}

    CalcArgs(int b, double u, double epsilon, double r_tst, double v_rms, double v_mean, int start, int stop, Eigen::Ref<Array> ax, Eigen::Ref<Array> ay, 
            Eigen::Ref<Array> az, ParticleSet *particles, ParticleSet *tracers) : block_size(b), mu(u), epsilon(epsilon), r_tst(r_tst), v_rms(v_rms), v_mean(v_mean)
                                                               , ax(ax.data(),ax.size()), ay(ay.data(),ay.size()), az(az.data(),az.size())
    {
        CalcArgs::start = start;
        CalcArgs::stop = stop;
        CalcArgs::particles = particles;
        CalcArgs::tracers = tracers;
    }
    
    void init(int b, double u, double epsilon, double r_tst, double v_rms, double v_mean, int start, int stop, Eigen::Ref<Array> iax, Eigen::Ref<Array> iay, 
            Eigen::Ref<Array> iaz, ParticleSet *particles, ParticleSet *tracers)
    {
        CalcArgs::block_size = b;
        CalcArgs::mu = u;
        CalcArgs::epsilon = epsilon;
        CalcArgs::r_tst = r_tst;
        CalcArgs::v_rms = v_rms;
        CalcArgs::v_mean = v_mean;
        new (&ax) ArrayRef(iax.data(), iax.size());
        new (&ay) ArrayRef(iay.data(), iay.size());
        new (&az) ArrayRef(iaz.data(), iaz.size());
        CalcArgs::start = start;
        CalcArgs::stop = stop;
        CalcArgs::particles = particles;
        CalcArgs::tracers = tracers;    
    }
};

class fof_arg
{
    public:
        ParticleSet *tracers;
        std::map<int,std::list<std::pair<int,std::array<double,7>>>> *match_dict;
        int i_min;
        int i_max;
        double timestep;

        void init(ParticleSet *tracers, std::map<int,std::list<std::pair<int,std::array<double,7>>>> *match_dict, double timestep, int i_min, int i_max)
        {
            fof_arg::tracers = tracers;
            fof_arg::match_dict = match_dict;
            fof_arg::timestep = timestep;
            fof_arg::i_min = i_min; 
            fof_arg::i_max = i_max; 
        }
};
std::vector<fof_arg> fof_args(3);

std::mutex tr_mutex;
std::mutex p_mutex;
std::mutex fof_mutex;
std::array<std::atomic<bool>,8> kill = {false,false,false,false,false,false,false,false};
std::array<std::condition_variable,8> cvar;
CalcArgs glob_args, glob_tr_args, glob_tr_args2;
std::array<std::mutex,8> mutexes;
bool ready[8] = {false,false,false,false,false,false,false,false};
bool processed[8] = {false, false, false, false, false,false,false,false};


double get_timestep(double v_mean, double r_tst, double v_rms, double a_max);
void create_particle_distribution(ParticleSet &particles, ParticleSet &tracers, double mu);
void kick_pos(ParticleSet &particles, ParticleSet &tracers, double &timestep);
void kick_vel(ParticleSet &particles, ParticleSet &tracers, double &timestep, double mu, double epsilon, int block_size,
        std::thread& calc1, std::thread& calc2, std::thread& calc3, std::thread& calc4, std::thread& calc5);
void write_output(std::string filename, ParticleSet &particles, ParticleSet &tracers);
void kick_tracers(ParticleSet &particles, ParticleSet &tracers, double &timestep, double mu, double epsilon, double block_size);
double init_timestep(ParticleSet &particles, double epsilon);
ParticleSet aggregate_tracers(ParticleSet &tracers, double link_length, double timestep, std::thread &fof_thr1, std::thread &fof_thr2, std::thread &fof_thr3);
void mat_section(int mutex_idx);
void lower_tri();
void block_diag();
void top_corner();
void signal(std::thread &thr, int idx);
void fof(int thrd_idx, fof_arg *args_addr);

namespace po = boost::program_options;

bool verbose = false;
double ttol = 0.01;
double k_deg = 0.2;
double r_lo = 0;
double r_hi = 1000;
double kTG = 1;
double dl_min = r_hi/1000;
double box_size; 
double end_time;
double link_length;

int main(int argc, char** argv)
{
    ParticleSet particles, tracers;
    double current_time = 0, timestep, mu, epsilon, next_write_time;
    int block_size, N_particles, numfiles, write_cntr=1;
    std::string filebase, outdir;
    std::ostringstream outfile;
    
    //read in user input and set variables
    po::options_description desc("compute velocity field decomposition with given basis");
    desc.add_options()
        ("help,h", "Print this help message")
        ("outdir",       po::value<std::string>(&outdir)->default_value("output/"), "output directory")
        ("outfile,o",    po::value<std::string>(&filebase)->default_value("test_grav"), "outputfilename")
        ("duration,T",   po::value<double>(&end_time)->default_value(1e1), "duration of simulation [0.98Myr]")
        //("timestep,t",   po::value<double>(&timestep)->default_value(1e-1), "timestep")
        ("mu,u",         po::value<double>(&mu)->default_value(1), "gravitational parameter for particles [pc(km/s)^2  (G=4.3e-3 [pc(km/s)^2/M0])")
        ("epsilon,e",    po::value<double>(&epsilon)->default_value(1e-2), "gravitational softening parameter")
        ("block_size,b", po::value<int>(&block_size)->default_value(300), "blocking size for performance (number of elements to group)")
        ("numfiles,f",   po::value<int>(&numfiles)->default_value(600), "number of files to write out")
        ("linklen,l",    po::value<double>(&link_length)->default_value(1e1), "linking length for tracer aggregation")
        ("verbose,v",    "whether to print verbose output")
        ("ttol",         po::value<double>(&ttol)->default_value(0.01), "tolerance for timestep")
        ("deg",          po::value<double>(&k_deg)->default_value(0.2), "order of power for density construction")
        //("r_lo",         po::value<double>(&r_lo)->default_value(1e-1), "lowest radius for density generation [pc]")
        //("r_hi",         po::value<double>(&r_hi)->default_value(1e6), "biggest radius for density generation [pc]")
        ("kTG",          po::value<double>(&kTG)->default_value(100),  "sigma of velocity for distribution generation, also interpreble as temp, k = 6.94e-60 [(km/s)^2*M0/K]")
        ("dl_min",       po::value<double>(&dl_min)->default_value(1e3/5000), "min update distance for particle with average speed")
        ("bx",           po::value<double>(&box_size)->default_value(1e3), "bounding box for periodic boundary conditions")
        ("nparticles,n", po::value<int>(&N_particles)->default_value(4000), "number of particles for the simulation"); 

    po::variables_map vm;
    try
    {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    }
    catch (po::error& e)
    { 
        std::cout<<"Option error: "<< e.what() <<std::endl;
        std::cout << std::string(60,'-') << std::endl;
        std::cout << desc << std::endl;  
        exit(-1);  
    }

    if(vm.count("help"))
    {
        std::cout<<desc<<std::endl;
        exit(1);
    }
    if(vm.count("verbose"))
    {
        verbose=true;
    }

    ////////////////run main code///////////////////
    particles = ParticleSet(N_particles, mu);
    tracers = ParticleSet(N_particles, mu);
    create_particle_distribution(particles, tracers, mu);
    timestep = init_timestep(particles, epsilon);
    next_write_time = end_time/(numfiles-1);
    outfile<<outdir<<filebase<<"_"<<write_cntr<<".hdf5";
    write_output(outfile.str(), particles, tracers);
    write_cntr++;
    outfile.str("");
    outfile.clear();
    
    //start threads
    std::thread calc1(top_corner);
    std::thread calc2(block_diag);
    std::thread calc3(lower_tri); 
    std::thread calc4(mat_section, 3);
    std::thread calc5(mat_section, 4);
    std::thread fof_thr1(fof, 5, &fof_args[0]);
    std::thread fof_thr2(fof, 6, &fof_args[1]);
    std::thread fof_thr3(fof, 7, &fof_args[2]);

    //start writing thread
    //std::thread(&write_thread, write_output_thread, write_buff);
    while(current_time < end_time || write_cntr<=numfiles)
    {
        kick_pos(particles, tracers, timestep);
        //kick_vel(particles, tracers, timestep, mu, epsilon, block_size, calc1, calc2, calc3, calc4, calc5);
        tracers = aggregate_tracers(tracers, link_length, timestep, fof_thr1, fof_thr2, fof_thr3);
        current_time += timestep;  
        if(current_time>=next_write_time)
        {
            outfile<<outdir<<filebase<<"_"<<write_cntr<<".hdf5";
            write_output(outfile.str(), particles, tracers);
            if(write_cntr%10==0)
                std::cout<<"write counter: "<<write_cntr<<std::endl;
            //obtain output FIFO queue mutex, add entry and move on if not too big
            //other wise wait for the size to decrease (can even do your own writing
            //until is small enough, but would obv have to drop after taking chunk)
            //once FIFO queue is small enough to keep moving, add to queue
            
            write_cntr++;
            outfile.str("");
            outfile.clear();
            next_write_time = current_time + end_time/(numfiles-1);
        }
    }
    
    for(auto &k : kill)
        k=true;

    signal(calc1, 0);
    signal(calc2, 1);
    signal(calc3, 2);
    signal(calc4, 3);
    signal(calc5, 4);
    signal(fof_thr1, 5);
    signal(fof_thr2, 6);
    signal(fof_thr3, 7);
    calc1.join();
    calc2.join();
    calc3.join();
    calc4.join();
    calc5.join();
    fof_thr1.join();
    fof_thr2.join();
    fof_thr3.join();

    std::cout<<"done."<<std::endl;
    return 0;
}


double init_timestep(ParticleSet &particles, double epsilon)
{
    int N_particles = particles.size;
    double x,y,z, dx, dy, dz, mu, A, v_rms=0, v_mean=0, a_max=-1e64, r_tst=1e46;
    Eigen::ArrayXd ax=Eigen::ArrayXd::Zero(N_particles), ay=Eigen::ArrayXd::Zero(N_particles), az=Eigen::ArrayXd::Zero(N_particles);

    for(int i=0 ;i<N_particles; i++)
    {

        v_rms += std::pow(particles.vel(i,0),2) + std::pow(particles.vel(i,1),2) +std::pow(particles.vel(i,2), 2);
        v_mean += std::sqrt( std::pow(particles.vel(i,0),2) + std::pow(particles.vel(i,1),2) +std::pow(particles.vel(i,2), 2) );
    }
    
    v_rms = v_rms/N_particles;
    v_mean = v_mean/N_particles;
    r_tst = 2 * box_size / std::pow(N_particles,0.333);
    
    return end_time/(600*4);  
}

double get_timestep(double v_mean, double r_tst, double v_rms, double a_max)
{
    return std::max(dl_min/v_mean, std::min(r_tst*ttol/2/v_mean, v_rms/a_max));
}

void kick_pos(ParticleSet &particles, ParticleSet &tracers, double &timestep)
{
    int N_particles = particles.size, N_tracers = tracers.size;
    for (int i=0; i<N_particles; i++)
    {
        particles.pos(i,0) = std::fmod(std::fmod(particles.pos(i,0) + timestep*particles.vel(i,0) + box_size/2, box_size) - box_size, box_size) + box_size/2;
        particles.pos(i,1) = std::fmod(std::fmod(particles.pos(i,1) + timestep*particles.vel(i,1) + box_size/2, box_size) - box_size, box_size) + box_size/2;
        particles.pos(i,2) = std::fmod(std::fmod(particles.pos(i,2) + timestep*particles.vel(i,2) + box_size/2, box_size) - box_size, box_size) + box_size/2;
    }
    for(int i=0; i<N_tracers; i++)
    {
        tracers.pos(i,0) = std::fmod(std::fmod(tracers.pos(i,0) + timestep*tracers.vel(i,0) + box_size/2, box_size) - box_size, box_size) + box_size/2;
        tracers.pos(i,1) = std::fmod(std::fmod(tracers.pos(i,1) + timestep*tracers.vel(i,1) + box_size/2, box_size) - box_size, box_size) + box_size/2;
        tracers.pos(i,2) = std::fmod(std::fmod(tracers.pos(i,2) + timestep*tracers.vel(i,2) + box_size/2, box_size) - box_size, box_size) + box_size/2;
    }
    return;
}

void signal(std::thread &thr, int idx)
{
    //signal threads to start calc
    std::lock_guard lk(mutexes[idx]);
    ready[idx] = true;
    cvar[idx].notify_one();
}
void wait(std::thread & thr, int idx)
{
    //wait for worker
    std::unique_lock lk(mutexes[idx]);
    cvar[idx].wait(lk, [idx]{return processed[idx];});
    processed[idx] = false;
    lk.unlock();
}

void kick_vel(ParticleSet &particles, ParticleSet &tracers, double &timestep, double mu, double epsilon, int block_size,
         std::thread & calc1, std::thread& calc2, std::thread& calc3, std::thread& calc4, std::thread& calc5)
{ 
    int N_particles = particles.size, N_tracers = tracers.size;
    double A, dx, dy, dz, dvx, dvy, dvz, x ,y ,z, v_rms=0, v_mean=0, a_max=-1e46, r_tst=1e64;
    Array ax=Array::Zero(N_particles), ay=Array::Zero(N_particles), az=Array::Zero(N_particles);
    Array tr_ax=Array::Zero(N_tracers), tr_ay=Array::Zero(N_tracers), tr_az=Array::Zero(N_tracers); 
    
    glob_args.init(   block_size, mu, epsilon, r_tst, v_rms, v_mean, NULL,                      NULL,                      ax,    ay,    az,    &particles, NULL);
    glob_tr_args.init(NULL,       mu, epsilon, NULL,  NULL,  NULL,   0,                         std::floor(N_particles/2), tr_ax, tr_ay, tr_az, &particles, &tracers);
    glob_tr_args2.init(NULL,      mu, epsilon, NULL,  NULL,  NULL,   std::floor(N_particles/2), N_particles,               tr_ax, tr_ay, tr_az, &particles, &tracers);
     
    signal(calc1, 0);
    signal(calc2, 1);
    signal(calc3, 2);
    
    signal(calc4, 3);
    signal(calc5, 4);

    wait(calc1, 0);
    wait(calc2, 1);
    wait(calc3, 2);
    
    v_rms = std::sqrt(glob_args.v_rms/N_particles);
    v_mean = glob_args.v_mean/N_particles;
    r_tst = glob_args.r_tst;
    a_max = (ax.square()+ay.square()+az.square()).sqrt().maxCoeff();
    timestep = get_timestep(v_mean, r_tst, v_mean, a_max); 
    
    if(verbose){
        std::cout<<"timestep: "<<timestep<<std::endl; 
        std::cout<<"v_mean: "<<v_mean<<", v_rms: "<<v_rms<<std::endl;
        std::cout<<"ax: "<<ax.abs().maxCoeff()<<", ay: "<<ay.abs().maxCoeff()<<", az: "<<az.abs().maxCoeff()<<std::endl;
        std::cout<<"r_tst: "<<r_tst<<std::endl;
    }

    for(int i=0; i<N_particles; i++)
    {
        particles.vel(i,0) += ax(i)*timestep;
        particles.vel(i,1) += ay(i)*timestep;
        particles.vel(i,2) += az(i)*timestep;
    }
    
    wait(calc4, 3);
    wait(calc5, 4);

    for(int i=0; i<N_tracers; i++)
    {
        tracers.vel(i,0) += tr_ax(i)*timestep;
        tracers.vel(i,1) += tr_ay(i)*timestep;
        tracers.vel(i,2) += tr_az(i)*timestep;
    }
}

void create_particle_distribution(ParticleSet &particles, ParticleSet &tracers, double mu)
{
    int N_particles = particles.size, N_tracers = tracers.size, cnt = 0;
    double r, phi, theta, a,  pmin, pmax;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> N(0, mu!=0 ? std::sqrt(kTG/mu) : std::sqrt(kTG));    
    a =  (std::pow(r_hi,k_deg+1) - std::pow(r_lo,k_deg+1) ) / (k_deg+1);
    pmin = a*std::pow(r_lo, 1/k_deg);
    pmax = a*std::pow(r_hi, 1/k_deg);
    std::uniform_real_distribution<> uniform(-box_size/2.0, box_size/2.0); //(pmin, pmax);

    for(int cnt=0; cnt<N_particles; cnt++) 
    {
        //r = std::pow(uniform(gen)/a, k_deg); 
        //phi = uniform(gen)* 2*3.14159265 / (pmax-pmin);
        //theta = std::acos(2*uniform(gen) / (pmax-pmin) - 1);
         
        particles.pos(cnt,0) = uniform(gen); //r*std::sin(theta)*std::cos(phi);
        particles.pos(cnt,1) = uniform(gen); //r*std::sin(theta)*std::sin(phi);
        particles.pos(cnt,2) = uniform(gen); //r*std::cos(theta);
        
        particles.vel(cnt,0) = N(gen); 
        particles.vel(cnt,1) = N(gen); 
        particles.vel(cnt,2) = N(gen); 

        tracers.pos(cnt,0) = particles.pos(cnt,0);
        tracers.pos(cnt,1) = particles.pos(cnt,1);
        tracers.pos(cnt,2) = particles.pos(cnt,2);
        tracers.vel(cnt,0) = particles.vel(cnt,0);
        tracers.vel(cnt,1) = particles.vel(cnt,1);
        tracers.vel(cnt,2) = particles.vel(cnt,2);
    }
    return;
}

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

void write_output(std::string filename, ParticleSet &particles, ParticleSet &tracers)
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

    //create datasets 
    create_parent_group(h5file, "particles/pos");
    H5::DataSet pos_dataset = h5file.createDataSet("particles/pos", H5::PredType::NATIVE_DOUBLE, file_dspace, H5P_DEFAULT);
    H5::DataSet vel_dataset = h5file.createDataSet("particles/vel", H5::PredType::NATIVE_DOUBLE, file_dspace, H5P_DEFAULT);
    H5::DataSet mu_dataset = h5file.createDataSet("particles/mu", H5::PredType::NATIVE_DOUBLE, mu_dspace, H5P_DEFAULT);

    create_parent_group(h5file, "tracers/pos");
    H5::DataSet tr_pos_dataset = h5file.createDataSet("tracers/pos", H5::PredType::NATIVE_DOUBLE, tr_file_dspace, H5P_DEFAULT);
    H5::DataSet tr_vel_dataset = h5file.createDataSet("tracers/vel", H5::PredType::NATIVE_DOUBLE, tr_file_dspace, H5P_DEFAULT); 
    H5::DataSet tr_mu_dataset = h5file.createDataSet("tracers/mu", H5::PredType::NATIVE_DOUBLE, tr_mu_dspace, H5P_DEFAULT);

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

    h5file.close();
    return;
}

//append ms onto m_list if ms.first is not equal to calling_idx or already in m_list
void unique_insert(int calling_idx, std::list<std::pair<int,std::array<double,7>>> &m_list, std::pair<int,std::array<double,7>> &ms)
{
    int p_idx = ms.first; 
    
    if(p_idx != calling_idx)
    {
        for(auto it=m_list.begin(); it!=m_list.end(); ++it)
        {
            if(it->first==p_idx)
                return;
        }   
        m_list.push_back(ms);
    }
}


ParticleSet aggregate_tracers(ParticleSet &tracers, double link_length, double timestep, std::thread &fof_thr1, std::thread &fof_thr2, std::thread &fof_thr3)
{
    Eigen::Array3d x1, x2, v1, v2, delta_x, delta_v;
    int N_tracers = tracers.size, p_idx; 
    double dotxv, dmin;
    Tensor_3D<double> pos,vel;
    std::vector<double> mu;
    std::map<int,std::list<std::pair<int,std::array<double,7>>>> match_dict;
   
    //initialize match_dict
    for (int i=0; i<N_tracers; i++)
    {
        match_dict[i] = std::list<std::pair<int,std::array<double,7>>>();
    }

    //split the tracer indices into groups for each thread
    int split[4];
    int r = N_tracers % 3;
    split[0] = 0;
    for(int i=1;i<=3;i++)
        split[i] = split[i-1] + int(N_tracers/3) + int(i<=r); 
    split[0] = 1;
 
    fof_args.at(0).init(&tracers, &match_dict, timestep, split[0], split[1]);
    fof_args.at(1).init(&tracers, &match_dict, timestep, split[1], split[2]);
    fof_args.at(2).init(&tracers, &match_dict, timestep, split[2], split[3]);

    signal(fof_thr1, 5);
    signal(fof_thr2, 6);
    signal(fof_thr3, 7);
  
    wait(fof_thr1, 5);
    wait(fof_thr2, 6);
    wait(fof_thr3, 7);

    //now reduce the data structure of groups to the individual groups
    for( auto &[calling_idx, m_list] :  match_dict)
    {
        //calling_idx is the particle whose matches we are examining
        //m_list is list of indices and pos/vels for the particles
        //matching the calling one
        for(auto & m : m_list)
        {
            //m is a particular pair of index and pos/vel
            p_idx = m.first;
            for(auto & ms : match_dict[p_idx])//insert these matches in calling list
            { 
                unique_insert(calling_idx, m_list, ms);
            }
            //now remove this particles group from the calling list
            //(since it will be accounted for in this loop call as
            //it's elements will now be checked while looping through
            //m_list)
            match_dict.erase(p_idx);
            //tracers.erase(p_idx); //not currently implemented
        }
    }

    //match dict should now be list of unique groups. combine these groups and have a new set of
    //aggregated particles
    double x,y,z,vx,vy,vz,u,mutot;
    pos = Tensor_3D<double>(1,match_dict.size(),3);
    vel = Tensor_3D<double>(1,match_dict.size(),3);
    int cnt = 0;
    for(auto &[p_idx, group_list] : match_dict)
    {
        x=0;
        y=0;
        z=0;
        vx=0;
        vy=0;
        vz=0;
        mutot=0;
       
        u = tracers.mu.at(p_idx);
        x += u*tracers.pos(p_idx,0);
        y += u*tracers.pos(p_idx,1);
        z += u*tracers.pos(p_idx,2);
        vx += u*tracers.vel(p_idx,0);
        vy += u*tracers.vel(p_idx,1);
        vz += u*tracers.vel(p_idx,2);
        mutot += u;
        for(auto &[idx,p] : group_list)
        {
            u = p[6];
            x += u*p[0];
            y += u*p[1];
            z += u*p[2];
            vx += u*p[3];
            vy += u*p[4];
            vz += u*p[5];
            mutot += u;
        }
        pos(cnt,0) = x/mutot;
        pos(cnt,1) = y/mutot;
        pos(cnt,2) = z/mutot;
        vel(cnt,0) = vx/mutot;
        vel(cnt,1) = vy/mutot;
        vel(cnt,2) = vz/mutot;
        mu.push_back(mutot);
        cnt++;
    }
    return ParticleSet(mu, pos, vel);
}

double impact_parameter(double &mu1, double &mu2, Eigen::Array3d & vrel, Eigen::Array3d &v1, Eigen::Array3d &v2)
{
    return 3.14159*(std::sqrt(mu1) + std::sqrt(mu2))/std::sqrt(vrel.square().sum()); // * std::pow((v1.square().sum() +v2.square().sum())/2 / vrel.square().sum(), 0.25);
}

void fof(int thrd_idx, fof_arg *args_addr)
{
    Eigen::Array3d x1, x2, v1, v2, delta_x ,delta_v;
    int N_tracers, p_idx; 
    double dotxv, dmin, t_crit;
    Tensor_3D<double> pos, vel;
    std::map<int,std::list<std::pair<int,std::array<double,7>>>> temp_match_dict;
    
    
    while(true)
    {
        //acquire lock and wait for data
        std::unique_lock lk(mutexes[thrd_idx]);
        cvar[thrd_idx].wait(lk, [thrd_idx]{return ready[thrd_idx];});
        ready[thrd_idx] = false;

        if(kill[thrd_idx])
            return;

        fof_arg args = *args_addr;
        N_tracers = args.tracers->size;
        temp_match_dict.clear(); 
        for (int i=0; i<N_tracers; i++)
        {
            temp_match_dict[i] = std::list<std::pair<int,std::array<double,7>>>();
        }

        //create list of maps which identify each particle with a particle index
        for(int i=args.i_min; i<args.i_max; i++)
        {
            x1 = {args.tracers->pos(i,0), args.tracers->pos(i,1), args.tracers->pos(i,2)};
            v1 = {args.tracers->vel(i,0), args.tracers->vel(i,1), args.tracers->vel(i,2)};
            for(int j=0; j<i; j++)
            {
                // do velocity based collision comparison
                x2 = {args.tracers->pos(j,0), args.tracers->pos(j,1), args.tracers->pos(j,2)};
                v2 = {args.tracers->vel(j,0), args.tracers->vel(j,1), args.tracers->vel(j,2)};
                delta_x = x1-x2;
                delta_v = v1-v2;
                dotxv = delta_x.cwiseProduct(delta_v).sum();
                t_crit = std::min(std::abs(dotxv)/delta_v.square().sum(), args.timestep);
                dmin = std::sqrt(delta_x.square().sum() + 2*dotxv*t_crit + delta_v.square().sum()*std::pow(t_crit,2));
                if(dotxv<0 and dmin<link_length and dmin<impact_parameter(args.tracers->mu.at(i), args.tracers->mu.at(j), delta_v, v1, v2))
                {
                    temp_match_dict[i].push_back(std::make_pair(j, std::array<double,7>({x2(0),x2(1),x2(2),v2(0),v2(1),v2(2), args.tracers->mu.at(j)}) ));
                    temp_match_dict[j].push_back(std::make_pair(i, std::array<double,7>({x1(0),x1(1),x1(2),v1(0),v1(1),v1(2), args.tracers->mu.at(i)}) ));
                }
            }
        }
            
        //obtain mutex and add entries from this thread to global match_dict
        fof_mutex.lock();
        for (auto &[p_idx, mtch_list] : temp_match_dict)
        { 
            for (auto & mtch : mtch_list)
            {
                //for now don't check this already exists b/c each thread looks at mutually excluive pairs, and match_dict
                //is renewed every time; so this pair can't already exist. 
                (*args.match_dict)[p_idx].push_back(mtch);
            }
        }
        fof_mutex.unlock(); 
         
        //signal that are done and return
        processed[thrd_idx] = true;
        lk.unlock();
        cvar[thrd_idx].notify_one(); 
    }
}


void top_corner()
{
    CalcArgs* args = &glob_args;
    double r_tst=1e64, x,y,z,dx,dy,dz,A, mu, epsilon, v_rms=0, v_mean=0, v2;
    ParticleSet *particles;
    int N_particles, block_size;
    Array ax_tmp, ay_tmp, az_tmp;
            
    while(true)
    {
        //acquire lock and wait for data
        std::unique_lock lk(mutexes[0]);
        cvar[0].wait(lk, []{return ready[0];});
        ready[0] = false;

        if(kill[0])
            return;

        r_tst=1e64;
        v_rms=0;
        v_mean = 0;
        particles = args->particles;
        N_particles = particles->size;
        block_size= args->block_size;
        mu = args->mu;
        epsilon = args->epsilon;
        ax_tmp=Array::Zero(N_particles);
        ay_tmp=Array::Zero(N_particles); 
        az_tmp=Array::Zero(N_particles); 

        v2 = std::pow(particles->vel(0,0),2) + std::pow(particles->vel(0,1),2) + std::pow(particles->vel(0,2),2);
        v_rms += v2;
        v_mean += std::sqrt(v2);
        for(int i=1; i<=block_size-1; i++)
        {
            x = particles->pos(i,0);
            y = particles->pos(i,1);
            z = particles->pos(i,2);
            v2 = std::pow(particles->vel(i,0),2) + std::pow(particles->vel(i,1),2) + std::pow(particles->vel(i,2),2);
            v_rms += v2;
            v_mean  += std::sqrt(v2);

            for(int j=0; j<i; j++)
            {
                dx = x - particles->pos(j,0);
                dy = y - particles->pos(j,1);
                dz = z - particles->pos(j,2);
                r_tst = std::min(std::sqrt(std::pow(dx,2) + std::pow(dy,2) +std::pow(dz,2)), r_tst);

                A = mu / (std::pow(std::pow(dx,2) + std::pow(dy,2) + std::pow(dz,2), 1.5) + epsilon); 
                ax_tmp(j) += A*dx;
                ay_tmp(j) += A*dy;
                az_tmp(j) += A*dz;

                ax_tmp(i) -= A*dx; 
                ay_tmp(i) -= A*dy;
                az_tmp(i) -= A*dz;          
            }
        }

        //get mutex to update accelerations
        p_mutex.lock();
        for(int i=0; i<N_particles;i++)
        {
            (args->ax)(i) += ax_tmp(i);
            (args->ay)(i) += ay_tmp(i);
            (args->az)(i) += az_tmp(i);
        }
        args->r_tst = std::min(r_tst, (args->r_tst));
        args->v_rms += v_rms; 
        args->v_mean += v_mean;
        p_mutex.unlock();

        //signal that are done and return
        processed[0] = true;
        lk.unlock();
        cvar[0].notify_one();
    }
}


void block_diag()
{
    CalcArgs* args = &glob_args;
    //now calculate block_bytes wide band so that you're preserving locality
    double r_tst=1e64, x,y,z,dx,dy,dz,A, mu, epsilon, v_rms=0, v_mean=0, v2;
    ParticleSet *particles;
    int N_particles, block_size;
    Array ax_tmp, ay_tmp, az_tmp;

    while(true)
    {
        //acquire lock and wait for data
        std::unique_lock lk(mutexes[1]);
        cvar[1].wait(lk, []{return ready[1];});
        ready[1] = false;

        if(kill[1])
            return;           

        r_tst=1e64;
        v_rms=0;
        v_mean = 0;
        particles = args->particles;
        N_particles = particles->size;
        block_size= args->block_size;
        mu = args->mu;
        epsilon = args->epsilon;
        ax_tmp = Array::Zero(N_particles);
        ay_tmp = Array::Zero(N_particles); 
        az_tmp = Array::Zero(N_particles); 
        
        for (int i=block_size;i<N_particles; i++)
        {
            x = particles->pos(i,0);
            y = particles->pos(i,1);
            z = particles->pos(i,2);
            v2 = std::pow(particles->vel(i,0),2) + std::pow(particles->vel(i,1),2) + std::pow(particles->vel(i,2),2);
            v_rms += v2;
            v_mean += std::sqrt(v2);
            for (int j=i-block_size+1; j<i; j++)
            {
                dx = x - particles->pos(j,0);
                dy = y - particles->pos(j,1);
                dz = z - particles->pos(j,2);        
                r_tst = std::min(std::sqrt(std::pow(dx,2) + std::pow(dy,2) +std::pow(dz,2)), r_tst);

                A = mu / (std::pow(std::pow(dx,2) + std::pow(dy,2) + std::pow(dz,2), 1.5) + epsilon); 
                ax_tmp(j) += A*dx;
                ay_tmp(j) += A*dy;
                az_tmp(j) += A*dz;

                ax_tmp(i) -= A*dx; 
                ay_tmp(i) -= A*dy;
                az_tmp(i) -= A*dz;    
            }
        }

        //get mutex to update accelerations
        p_mutex.lock();
        for (int i=0; i<N_particles;i++)
        {
            (args->ax)(i) += ax_tmp(i);
            (args->ay)(i) += ay_tmp(i);
            (args->az)(i) += az_tmp(i);
        }
        args->r_tst = std::min(r_tst, (args->r_tst));
        args->v_rms += v_rms;
        args->v_mean += v_mean;
        p_mutex.unlock();

        //signal that are done and return
        processed[1] = true;
        lk.unlock();
        cvar[1].notify_one();
    }
}


void lower_tri()
{
    CalcArgs* args = &glob_args;
    double r_tst=1e64, x,y,z,dx,dy,dz,A, mu, epsilon;
    ParticleSet *particles;
    int N_particles, block_size;
    Array ax_tmp, ay_tmp, az_tmp;
    
    while(true)
    {
        //acquire lock and wait for data
        std::unique_lock lk(mutexes[2]);
        cvar[2].wait(lk, []{return ready[2];});
        ready[2]  = false;

        if(kill[2])
            return;

        r_tst=1e64;
        particles = args->particles;
        N_particles = particles->size;
        block_size= args->block_size;
        mu = args->mu;
        epsilon = args->epsilon;
        ax_tmp=Array::Zero(N_particles);
        ay_tmp=Array::Zero(N_particles); 
        az_tmp=Array::Zero(N_particles);

        //now fill in the rest
        for (int j=0; j<N_particles-block_size+1; j++)
        {
            x = particles->pos(j,0);
            y = particles->pos(j,1);
            z = particles->pos(j,2);
            for (int i=j+block_size; i<N_particles; i++)
            {
                dx = particles->pos(i,0) - x;
                dy = particles->pos(i,1) - y;
                dz = particles->pos(i,2) - z;
                r_tst = std::min(std::sqrt(std::pow(dx,2) + std::pow(dy,2) +std::pow(dz,2)), r_tst);

                A = mu / ( std::pow(std::pow(dx,2) + std::pow(dy,2) + std::pow(dz,2), 1.5) + epsilon); 
                ax_tmp(j) += A*dx;
                ay_tmp(j) += A*dy;
                az_tmp(j) += A*dz;

                ax_tmp(i) -= A*dx; 
                ay_tmp(i) -= A*dy;
                az_tmp(i) -= A*dz;    
            }
        }

        //get mutex to update accelerations
        p_mutex.lock();
        for (int i=0; i<N_particles; i++)
        {
            (args->ax)(i) += ax_tmp(i);
            (args->ay)(i) += ay_tmp(i);
            (args->az)(i) += az_tmp(i);
        }
        (args->r_tst) = std::min(r_tst, (args->r_tst));
        p_mutex.unlock();

        //signal that are done and return
        processed[2] = true;
        lk.unlock();
        cvar[2].notify_one();
    }
}

void mat_section(int mutex_idx)
{
    double  x,y,z,dx,dy,dz,A, mu, epsilon;
    ParticleSet *particles, *tracers;
    int N_particles, N_tracers, block_size, start, stop;
    Array ax_tmp, ay_tmp, az_tmp; 
    CalcArgs* args;
    if (mutex_idx==3) 
        args = &glob_tr_args;
    else
        args = &glob_tr_args2;

    while(true)
    {
        //acquire lock and wait for data
        std::unique_lock lk(mutexes[mutex_idx]);
        cvar[mutex_idx].wait(lk, [mutex_idx]{return ready[mutex_idx];});
        ready[mutex_idx] = false;
        
        if(kill[mutex_idx])
            return;

        particles = args->particles;
        tracers = args->tracers;
        N_particles = particles->size;
        N_tracers = tracers->size;
        start = args->start;
        stop = args->stop;
        block_size= args->block_size;
        mu = args->mu;
        epsilon = args->epsilon;
        ax_tmp=Array::Zero(N_tracers);
        ay_tmp=Array::Zero(N_tracers); 
        az_tmp=Array::Zero(N_tracers);

        //now fill in the rest
        for (int i=start; i<stop; i++)
        {
            x = particles->pos(i,0);
            y = particles->pos(i,1);
            z = particles->pos(i,2);
            for (int j=0; j<N_tracers; j++)
            {
                dx = x - tracers->pos(j,0);
                dy = y - tracers->pos(j,1);
                dz = z - tracers->pos(j,2);
              
                A = mu / ( std::pow(std::pow(dx,2) + std::pow(dy,2) + std::pow(dz,2), 1.5) + epsilon); 
                ax_tmp(j) += A*dx;
                ay_tmp(j) += A*dy;
                az_tmp(j) += A*dz; 
            }
        }
        
        //get mutex to update accelerations
        tr_mutex.lock();
        for (int i=0; i<N_tracers;i++)
        {
            (args->ax)(i) += ax_tmp(i);
            (args->ay)(i) += ay_tmp(i);
            (args->az)(i) += az_tmp(i);
        }
        tr_mutex.unlock();
        
        //signal that are done and return
        processed[mutex_idx] = true;
        lk.unlock();
        cvar[mutex_idx].notify_one();
    }
}
