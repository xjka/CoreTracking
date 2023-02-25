#define _USE_MATH_DEFINES
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


//simplest possible histogram for sequence of integers between 1 and 'max'
class Histogram
{
    public:
        std::vector<float> counts;
        int total_count;
        int maxbin;

        Histogram()
        {
            total_count = 0;
        }

        Histogram(int max) //biggest group size we expect to encounter
        {   
            maxbin = max;
            counts = std::vector<float>(max, 0);
            total_count = 0;
        }

        void add(int group_size){
            if(0<group_size and group_size<=maxbin){
                counts[group_size-1] += 1;
                total_count += group_size;
            }
            else if(0<group_size){
                counts.resize(group_size);
                std::fill(counts.begin()+maxbin, counts.end(), 0);
                maxbin = group_size;
                counts[group_size-1] += 1;
                total_count += group_size;
            }
        }
        void resize(int new_size){
            counts.resize(new_size);
        }
        void reset(){
            std::fill(counts.begin(), counts.end(), 0);
            total_count = 0; 
        }
};


std::string create_parent_group(H5::H5File &h5file, std::string group_name);
void write_output(std::string filename, ParticleSet &particles, ParticleSet &tracers, double &current_time, Histogram &hist);
void write_slab(H5::DataSet &dataset, Tensor_3D<double> &p, std::vector<hsize_t> &offset, H5::DataSpace &mem_dspace, H5::DataSpace &file_dspace );
void file_init(std::string filename, ParticleSet &particles, ParticleSet &tracers, double &current_time);
