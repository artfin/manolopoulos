#include <iostream>
#include <string>
#include <cmath>
#include <Eigen/Dense>
#include <algorithm>
#include <fstream>

#include "./log.hpp"

#define DEBUG_VALUE_OF(variable) { _logDebug.value_of(#variable, variable); } 
template<class T> void __log__::value_of(const std::string& name, const T& value) 
{
    fill_wsp();
    std::string tp = demangled_type_info_name(typeid(value));
    std::size_t found = tp.find("Eigen");
    if ( found != std::string::npos )
        std::cout << wsp << name << " (" << demangled_type_info_name(typeid(value)) << ") = \n" << value << std::endl;
    else
        std::cout << wsp << name << " (" << demangled_type_info_name(typeid(value)) << ") = [" << value << "]" << std::endl;
}

inline int delta(int i, int j) { return i == j; }

void potmat(Eigen::MatrixXd & w, const double r, const int nch)
// Example 5.2, p.114.
// The second test problem in coupled Schrodinger equations section.
// Study of special algorithms for solving Sturm-Liouville and Schroedinger equations
// Verrie Ledoux, Thesis
// The first few eigenvalues of the problem are:
// 14.94180054
// 17.04349658
// 21.38042053
// 26.92073133
// 51.82570724
// 55.80351609
{
    assert(nch == 4 && "Example expects nch=4");

    double cos_r = std::cos(r);

    for ( int i = 0; i < nch; ++i )
    {
        for ( int j = 0; j < nch; ++j )
        {
            w(i, j) = cos_r / std::max(i + 1, j + 1) + delta(i + 1, j + 1) / std::pow(r, i + 1);
        }
    }
}

void logdb(const double Energy, Eigen::MatrixXd & z, Eigen::MatrixXd & w, Eigen::MatrixXd & wref, const int nch, const double rmin, const double rmax, const int nsteps)
// routine to initialise the log derivative matrix, y, at r = rmin, 
// and propagate it from rmin to rmax 
// ----------------------------------------------------------
// variables in call list:
// Energy       
//
// z            matrix of dimension (nch, nch) 
//              on return `z` contains the log derivative matrix at r = rmax
// w            workspace for the potential matrix
// wref         workspace for the reference potential
// nch          number of coupled equations
// rmin, rmax   integration range limits
//              rmin is assummed to lie inside the classically forbidden
//              region for all channels
// nsteps       number of sectors partitioning integration range
//
// detZ         vector of determinant of log derivative matrix
{
    DEBUG_METHOD("logdb");

    double h = (rmax - rmin) / (2.0 * nsteps);
    DEBUG_VALUE_OF(h);

    double d3 = h / 3.0;
    double d6 = h * h / 6.0;

    double r = rmin;
    potmat(w, r, nch);
    DEBUG_VALUE_OF(w);

    // use diagonal approximation to WKB initial value for log derivative matrix
    // (equation 16)
    // rmin is assumed to lie inside classically forbidden region in all 
    // the channels (w(ii) > 0)
    //
    // zero out z matrix
    z = Eigen::MatrixXd::Zero(nch, nch);

    //double wdiag = 0.0;
    for ( int i = 0; i < nch; ++i )
    {
        //wdiag = w(i, i);
        //if ( wdiag < 0.0 )
        //{
            //std::cout << "DIAGONAL REFERENCE POTENTIAL SHOULD BE POSITIVE! ABORT." << std::endl;
            //exit( 1 );
        //}

        //z(i, i) = std::sqrt(wdiag);
        z(i, i) = 1.0e20;
    }
    
    Eigen::MatrixXd z1_ab = Eigen::MatrixXd::Zero(nch, nch);

    Eigen::MatrixXd z1_ac = Eigen::MatrixXd::Zero(nch, nch);
    Eigen::MatrixXd z2_ac = Eigen::MatrixXd::Zero(nch, nch);
    Eigen::MatrixXd z3_ac = Eigen::MatrixXd::Zero(nch, nch);
    Eigen::MatrixXd z4_ac = Eigen::MatrixXd::Zero(nch, nch);
    
    Eigen::MatrixXd z1_cb = Eigen::MatrixXd::Zero(nch, nch);
    Eigen::MatrixXd z2_cb = Eigen::MatrixXd::Zero(nch, nch);
    Eigen::MatrixXd z3_cb = Eigen::MatrixXd::Zero(nch, nch);
    Eigen::MatrixXd z4_cb = Eigen::MatrixXd::Zero(nch, nch);

    Eigen::MatrixXd Qa = Eigen::MatrixXd::Zero(nch, nch);
    Eigen::MatrixXd Qb = Eigen::MatrixXd::Zero(nch, nch);
    Eigen::MatrixXd Qc = Eigen::MatrixXd::Zero(nch, nch);

    Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(nch, nch);
    Eigen::MatrixXd z_old = Eigen::MatrixXd::Zero(nch, nch);
    Eigen::MatrixXd z_acb = Eigen::MatrixXd::Zero(nch, nch);
    

    std::ofstream ofs("det.txt");

    int istep = 1;
    // propagate z from rmin to rmax
    for ( int kstep = 1; kstep <= nsteps; ++kstep, istep += 2, z_old = z )
    {
        // the reference potential for the sector is the diagonal of the
        // coupling matrix evaluated at the centre of the sector, r = c
        // (eqn 15 of Manolopoulos1986)
        r = rmin + istep * h;
        DEBUG_VALUE_OF(r);
        DEBUG_PRINT("Reference potential is evaluated at the centre of the sector (r = c)\n"); 
        potmat(w, r, nch);
        
        // the reference potential:
        for ( int ich = 0; ich < nch; ++ich )
            wref(ich, ich) = w(ich, ich);

        // evaluate half sector propagators (eqn 10, manolopoulos1986)
        // z1: y1, z2: y2, z3: y3, z4: y4
        double arg;
        for ( int ich = 0; ich < nch; ++ich )
        {
            double pj2 = wref(ich, ich) - Energy;
            arg = std::sqrt(std::abs(pj2));
            
            if ( pj2 < 0.0 )
            {
                double ctn = 1.0 / std::tan(arg * h);
                z1_ac(ich, ich) = z1_cb(ich, ich) = arg * ctn;
                double csc = 1.0 / std::sin(arg * h);
                z2_ac(ich, ich) = z2_cb(ich, ich) = arg * csc;
            }
            else
            {
                double ctnh = 1.0 / std::tanh(arg * h);
                z1_ac(ich, ich) = z1_cb(ich, ich) = arg * ctnh;
                double csch = 1.0 / std::sinh(arg * h); 
                z2_ac(ich, ich) = z2_cb(ich, ich) = arg * csch;
            }

            z3_ac(ich, ich) = z3_cb(ich, ich) = z2_ac(ich, ich);
            z4_ac(ich, ich) = z4_cb(ich, ich) = z1_ac(ich, ich);
        } 
        //DEBUG_VALUE_OF(z1);
        //DEBUG_VALUE_OF(z2);

        // adjust quadrature contribution at r = a to account for
        // sector reference potential
        r = rmin + 2.0 * (kstep - 1) * h;
        potmat(w, r, nch);
        Qa = d3 * (w - wref);
        std::cout << "r: " << r << " for Qa evaluation" << std::endl;

        // evaluate quadrature contribution at midpoint r = c 
        // (eqn 12 from Manolopoulos1986)
        // evaluate potential matrix at midpoint r = c
        // матрица w в точке r = c была рассчитана в самом начале цикла
        // при расчете Wref
        // уже можно поганить эту матрицу, так что делаем inplace transformations  
        r = rmin + istep * h;
        potmat(w, r, nch);
        std::cout << "r: " << r << " for Qc evaluation" << std::endl;
        w *= -d6;
        // U(c) = w(c) - wref(c), т.к. wref вычисляется как диагональная часть w(c), то U(c) имеет нулевые элементы на диагонали
        // а значит I - U(c) содержит единицы на диагонали
        for ( int ich = 0; ich < nch; ++ich )
            w(ich, ich) = 1.0;
        // w now containts I - h^2/6*U(c)
        w = w.inverse();
        
        // subtracting identity matrix
        for ( int ich = 0; ich < nch; ++ich )
            w(ich, ich) -= 1.0;

        Qc = w * 4.0 / h;
        // w now contains Q(c)
       
        z1_ac += Qa;
        z4_ac += Qc;
        //DEBUG_VALUE_OF(z1);
        //DEBUG_VALUE_OF(z2);
        //DEBUG_VALUE_OF(z3);
        //DEBUG_VALUE_OF(z4);

        // propagate z matrix across the first half sector
        // from r = a to r = c (eqn 14 of Manolopoulos1986)
        
        // Y(c) = z4(a,c) - z3(a,c) * (z(a) + z1(a,c))^(-1) * z2(a,c)
        z += z1_ac;
        z = z.inverse();
        // z now contains (z(a) + z1(a,c))^(-1)
        z = z4_ac - z3_ac * z * z2_ac;
        // z now contains z(c)
       
        r = rmin + 2.0 * kstep * h;
        potmat(w, r, nch);
        Qb = d3 * (w - wref);
        std::cout << "r: " << r << " for Qb evaluation" << std::endl;

        z1_cb += Qc;
        z4_cb += Qb; 

        // propagate z matrix across the second half sector
        // from r = c to r = b (eqn 14 from Manolopoulos1986)
        
        // Y(b) = z4(c, b) - z3(c, b) * (z(c) + z1(c, b))^(-1) * z2(c, b)
        z += z1_cb;
        z = z.inverse();
        // z now contains (z(c) + z1(c, b))^(-1)
        z = z4_cb - z3_cb * z * z2_cb;
        // z now contains Y(b)

        z_acb = (z4_ac + z1_cb).inverse();
        z1_ab = z1_ac - z2_ac * z_acb * z3_ac;
        Z = z_old - z1_ab;

        r = rmin + 2.0 * (kstep - 1) * h;
        ofs << r << " " << Z.determinant() << std::endl;
        DEBUG_VALUE_OF(z);
    }
}

int main()
{
    DEBUG_METHOD("main");

    const int nch = 4;
    Eigen::MatrixXd z = Eigen::MatrixXd::Zero(nch, nch);
    Eigen::MatrixXd w = Eigen::MatrixXd::Zero(nch, nch);
    Eigen::MatrixXd wref = Eigen::MatrixXd::Zero(nch, nch);
    
    const double rmin = 0.1;
    const double rmax = 1.0;

    const int nsteps = 100;

    const double Energy = 53.0; 

    logdb(Energy, z, w, wref, nch, rmin, rmax, nsteps);
    
    return 0;
}
