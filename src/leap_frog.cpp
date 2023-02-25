#include "utilities.hpp"

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
void create_particle_distribution(ParticleSet &particles, ParticleSet &tracers, double mu, std::string &input_file, double &current_time);
void kick_pos(ParticleSet &particles, ParticleSet &tracers, double &timestep);
void kick_vel(ParticleSet &particles, ParticleSet &tracers, double &timestep, double mu, double epsilon, int block_size,
        std::thread& calc1, std::thread& calc2, std::thread& calc3, std::thread& calc4, std::thread& calc5);
void kick_tracers(ParticleSet &particles, ParticleSet &tracers, double &timestep, double mu, double epsilon, double block_size);
double init_timestep(ParticleSet &particles, double epsilon);
ParticleSet aggregate_tracers(ParticleSet &tracers, double link_length, double timestep, std::vector<std::thread> &fof_threads, Histogram &hist);
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
double box_size, end_time, link_length, b=1, tf=4;
int numfiles, num_agg_threads=3, num_calc_threads=5;

int main(int argc, char** argv)
{
    ParticleSet particles, tracers;
    double current_time = 0, timestep, mu, epsilon, next_write_time;
    int block_size, N_particles, write_cntr=1;
    std::string filebase, outdir, input_file;
    std::ostringstream outfile;
    Histogram hist;

    //read in user input and set variables
    po::options_description desc("compute velocity field decomposition with given basis");
    desc.add_options()
        ("help,h", "Print this help message")
        ("outdir",       po::value<std::string>(&outdir)->default_value("output/"), "output directory")
        ("outfile,o",    po::value<std::string>(&filebase)->default_value("test_grav"), "outputfilename")
        ("duration,T",   po::value<double>(&end_time)->default_value(50), "duration of simulation [0.98Myr]")
        ("infile,i",     po::value<std::string>(&input_file)->default_value(""), "file to intialize simulation from")
        ("tf,",          po::value<double>(&tf)->default_value(4), "timestep feactor for temporary maxwell distribution test")
        ("mu,u",         po::value<double>(&mu)->default_value(10), "gravitational parameter for particles [pc(km/s)^2  (G=4.3e-3 [pc(km/s)^2/M0])")
        ("epsilon,e",    po::value<double>(&epsilon)->default_value(1e-2), "gravitational softening parameter")
        ("b,b",          po::value<double>(&b)->default_value(1), "constant infront of impact parameter test")
        ("numfiles,f",   po::value<int>(&numfiles)->default_value(600), "number of files to write out")
        ("linklen,l",    po::value<double>(&link_length)->default_value(50), "linking length for tracer aggregation")
        ("verbose,v",    "whether to print verbose output")
        ("ttol",         po::value<double>(&ttol)->default_value(0.01), "tolerance for timestep")
        ("deg",          po::value<double>(&k_deg)->default_value(0.2), "order of power for density construction")
        //("r_lo",         po::value<double>(&r_lo)->default_value(1e-1), "lowest radius for density generation [pc]")
        //("r_hi",         po::value<double>(&r_hi)->default_value(1e6), "biggest radius for density generation [pc]")
        ("kTG",          po::value<double>(&kTG)->default_value(100),  "sigma of velocity for distribution generation, also interpreble as temp, k = 6.94e-60 [(km/s)^2*M0/K]")
        ("dl_min",       po::value<double>(&dl_min)->default_value(1e3/5000), "min update distance for particle with average speed")
        ("bx",           po::value<double>(&box_size)->default_value(1e3), "bounding box for periodic boundary conditions")
        ("nparticles,n", po::value<int>(&N_particles)->default_value(8000), "number of particles for the simulation"); 

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
    block_size = int(N_particles/2.0);
    particles = ParticleSet(N_particles, mu);
    tracers = ParticleSet(N_particles, mu);
    create_particle_distribution(particles, tracers, mu, input_file, current_time);
    timestep = init_timestep(particles, epsilon);
    next_write_time = end_time/(numfiles-1);
    outfile<<outdir<<filebase<<"_"<<write_cntr<<".hdf5";
    write_output(outfile.str(), particles, tracers, current_time, hist);
    write_cntr++;
    outfile.str("");
    outfile.clear();
    
    //start threads
    std::thread calc1(top_corner);
    std::thread calc2(block_diag);
    std::thread calc3(lower_tri); 
    std::thread calc4(mat_section, 3);
    std::thread calc5(mat_section, 4);
    std::vector<std::thread> fof_threads;
    for(int i=0;i<num_agg_threads; i++)
        fof_threads.push_back(std::thread (fof, num_calc_threads+i, &fof_args[i]));

    //start writing thread
    //std::thread(&write_thread, write_output_thread, write_buff);
    while(current_time < end_time || write_cntr<=numfiles)
    {
        kick_pos(particles, tracers, timestep);
        //kick_vel(particles, tracers, timestep, mu, epsilon, block_size, calc1, calc2, calc3, calc4, calc5);
        tracers = aggregate_tracers(tracers, link_length, timestep, fof_threads, hist);
        current_time += timestep;  
        if(current_time>=next_write_time)
        {
            outfile<<outdir<<filebase<<"_"<<write_cntr<<".hdf5";
            write_output(outfile.str(), particles, tracers, current_time, hist);
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
    for(int i=0; i<num_agg_threads; i++)
        signal(fof_threads.at(i), num_calc_threads+i);
    
    calc1.join();
    calc2.join();
    calc3.join();
    calc4.join();
    calc5.join();
    for(int i=0; i<num_agg_threads; i++)
        fof_threads.at(i).join();

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
    
    return end_time/(numfiles*tf);  
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

void create_particle_distribution(ParticleSet &particles, ParticleSet &tracers, double mu, std::string &input_file, double &current_time)
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
    
    //if an input file is specified initialize from that
    if(input_file != "")
    {
        std::cout<<"generating from input_file  (will ignore any conflicting cmd-line arguments)"<<std::endl;
        file_init(input_file, particles, tracers, current_time); 
    }
    else //else just do usual random generation of inital state
    {
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
    }
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

std::vector<int> thread_split(int numthreads, int N_elements)
{
    std::vector<int> split(numthreads+1), n_cmprs(numthreads);
        int r, tot_comparisons = (int) ((std::pow(N_elements,2) + N_elements)/2.0);
    r = tot_comparisons % numthreads;

    //first get the size of each split
    for(int i=0; i <numthreads; i++)
        n_cmprs[i] = int(tot_comparisons/numthreads) + int(i<r); 
    
    //now find best match of indices so that split of pairwise 
    //indices roughly follows this division of labor
    split[0] = 0;
    for(int i=1;i<numthreads;i++){
        split[i] = (int) std::round((std::sqrt(1 + 4*(2*n_cmprs[i-1] + std::pow(split[i-1],2) - split[i-1])) + 1) / 2.0);
        n_cmprs[i] += n_cmprs[i-1] - (std::pow(split[i],2)-split[i] - (std::pow(split[i-1],2) -split[i-1]) )/2.0;
    }
    split[numthreads] = N_elements; //the final split point simply must point to the end 
    split[0] = 1; //first should point to 1 (because we're looping over sub-diagonal)
    
    return split;
}

ParticleSet aggregate_tracers(ParticleSet &tracers, double link_length, double timestep, std::vector<std::thread> &fof_threads, Histogram & hist)
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
    std::vector<int> split = thread_split(num_agg_threads, N_tracers); 

    for(int j=0; j<num_agg_threads; j++)
        fof_args.at(j).init(&tracers, &match_dict, timestep, split[j], split[j+1]);
       
    for(int j=0; j<num_agg_threads; j++)
        signal(fof_threads.at(j), num_calc_threads+j);
    
    for(int j=0; j<num_agg_threads; j++)
        wait(fof_threads.at(j), num_calc_threads+j);

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
    hist.reset();
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
        if(group_list.size()>0){
            hist.add(group_list.size()+1);
        }
        else{
            hist.add(1);
        }
    }

    if(hist.total_count != N_tracers){
        std::cout<<"Uh-oh, number of particles in match_dict groups: "<<hist.total_count<<", ";
        std::cout<<"doesn't match the initial number of particles before this merge step: "<<N_tracers<<std::endl;
    }

    return ParticleSet(mu, pos, vel);
}

double impact_parameter(double &mu1, double &mu2, Eigen::Array3d & vrel, Eigen::Array3d &v1, Eigen::Array3d &v2)
{
    return b * std::sqrt( std::pow(mu1+mu2,3) / (mu1*mu2 * vrel.square().sum() * M_PI*M_SQRT2) ); 
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
