#include "ceres/ceres.h"
#include "gtest/gtest.h"

#include "SimpleBundleAdjuster.h"
#include "BundleAdjuster.h"

namespace {

// Step 2. Use the TEST macro to define your tests.
//
// TEST has two parameters: the test case name and the test name.
// After using the macro, you should define your test logic between a
// pair of braces.  You can use a bunch of macros to indicate the
// success or failure of a test.  EXPECT_TRUE and EXPECT_EQ are
// examples of such macros.  For a complete list, see gtest.h.
//
// <TechnicalDetails>
//
// In Google Test, tests are grouped into test cases.  This is how we
// keep test code organized.  You should put logically related tests
// into the same test case.
//
// The test case name and the test name should both be valid C++
// identifiers.  And you should not use underscore (_) in the names.
//
// Google Test guarantees that each test you define is run exactly
// once, but it makes no guarantee on the order the tests are
// executed.  Therefore, you should write your tests in such a way
// that their results don't depend on their order.
//
// </TechnicalDetails>

// inspired by: https://github.com/colmap/colmap/blob/main/src/colmap/estimators/bundle_adjustment_test.cc
TEST(BundleAdjusterTest, Basic)
{
    // BALProblem bal_problem;
    // std::string filename = "../Data/problem-49-7776-pre.txt";
    // if (!bal_problem.LoadFile(filename.c_str()))
    // {
    //     std::cerr << "Unable to open file: " << filename << "\n";
    //     exit(1);
    // }

    // const double* observations = bal_problem.observations();

    // // Create residuals for each observation in the bundle adjustment problem. 
    // // The parameters for cameras and points are added automatically
    // ceres::Problem problem;
    // for (int i = 0; i < bal_problem.num_observations(); ++i)
    // {
    //     // Each residual block takes a point and a cameras as input
    //     // and outputs a 2D residual. Internally, the cost function
    //     // stores the image location and compares the reprojection against the observation
    //     ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(
    //         observations[2*i+0],observations[2*i+1]);
    //     problem.AddResidualBlock(cost_function,
    //         nullptr /* squared loss */,
    //         bal_problem.mutable_camera_for_observation(i),
    //         bal_problem.mutable_point_for_observation(i));
    // }

    // // Make Ceres automatically detect the bundle structure. Note
    // // that the standard solver, SPARSE_NORMAL_CHOLESKY, also works fine
    // // but is slower for standard bundle adjustment problems.
    // ceres::Solver::Options options;
    // options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.minimizer_progress_to_stdout = true;

    // ceres::Solver::Summary summary;
    // ceres::Solve(options, &problem, &summary);
    // std::cout << summary.FullReport() << "\n";
}

TEST(BundleAdjusterTest, SmcBundleAdjuster)
{
    // BALProblem bal_problem;
    // std::string filename = "../Data/problem-49-7776-pre.txt";
    // if (!bal_problem.LoadFile(filename.c_str()))
    // {
    //     std::cerr << "Unable to open file: " << filename << "\n";
    //     exit(1);
    // }

    // const double* observations = bal_problem.observations();

    // BundleAdjuster bundleAdjuster;
    // bundleAdjuster.setupProblem();

    // for (int i = 0; i < bal_problem.num_observations(); ++i)
    // {
    //     // Each residual block takes a point and a cameras as input
    //     // and outputs a 2D residual. Internally, the cost function
    //     // stores the image location and compares the reprojection against the observation
    //     bundleAdjuster.addPointToProblem(
    //         observations[2*i+0],
    //         observations[2*i+1],
    //         bal_problem.mutable_camera_for_observation(i),
    //         bal_problem.mutable_point_for_observation(i)
    //     );
    // }

    // bundleAdjuster.solve();
}

#include "Utils.h"

TEST(ReprojectionError, ReprojectionError)
{
    std::vector<cv::Point2f> observed = 
    {
        {120, 200},
        {130, 210},
        {140, 220},
        {150, 230}
    };

    std::vector<cv::Point2f> projected =
    {
        {118, 198},
        {132, 212},
        {141, 219},
        {148, 228}
    };
    std::tuple<double, double> errors = calculateReprojectionError(observed, projected);
    std::cout << "mean: " << std::get<0>(errors) << ", rms: " << std::get<1>(errors) << "\n";
}

TEST(BundleAdjusterTest, SmcBundleAdjuster2)
{
    // BALProblem bal_problem;
    // std::string filename = "../Data/problem-49-7776-pre.txt";
    // if (!bal_problem.LoadFile(filename.c_str()))
    // {
    //     std::cerr << "Unable to open file: " << filename << "\n";
    //     exit(1);
    // }

    // const double* observations = bal_problem.observations();

    // BundleAdjuster bundleAdjuster;
    // bundleAdjuster.setupProblem();

    // std::map<double*, size_t> seen_cameras;
    // std::map<double*, size_t> seen_points;
    // size_t point_idx = 0;
    // for (int i = 0; i < bal_problem.num_observations(); ++i)
    // {
    //     double* camera = bal_problem.mutable_camera_for_observation(i);
    //     size_t camera_idx = 0;
    //     if (seen_cameras.find(camera) == seen_cameras.end())
    //     {
    //         double c[9] = {
    //             camera[0],camera[1],camera[2],
    //             camera[3],camera[4],camera[5],
    //             camera[6],camera[7],camera[8]
    //         };
    //         auto cIdx = bundleAdjuster.addCameraParam(c);
    //         seen_cameras.insert({camera, cIdx});
    //         camera_idx = cIdx;
    //     }
    //     else {
    //         camera_idx = seen_cameras.find(camera)->second;
    //     }
        
    //     double* point = bal_problem.mutable_point_for_observation(i);
    //     size_t point_idx = 0;
    //     if (seen_points.find(point) == seen_points.end())
    //     {
    //         double p[3] = {
    //             point[0],point[1],point[2]
    //         };
    //         auto pIdx = bundleAdjuster.addWorldPointParam(p);
    //         seen_points.insert({point, pIdx});
    //         point_idx = pIdx;
    //     }
    //     else {
    //         point_idx = seen_points.find(point)->second;
    //     }
    //     bundleAdjuster.addObservation(
    //         camera_idx,
    //         point_idx,
    //         camera_idx,
    //         point_idx,
    //         observations[2*i+0],
    //         observations[2*i+1]
    //     );
    //     // Each residual block takes a point and a cameras as input
    //     // and outputs a 2D residual. Internally, the cost function
    //     // stores the image location and compares the reprojection against the observation

    //     bundleAdjuster.addPointToProblem(
    //         observations[2*i+0],
    //         observations[2*i+1],
    //         bundleAdjuster.mutable_camera_for_observation(i),
    //         bundleAdjuster.mutable_point_for_observation(i)
    //     );
    // }

    // bundleAdjuster.solve();
}
}  // namespace

// Step 3. Call RUN_ALL_TESTS() in main().
//
// We do this by linking in src/gtest_main.cc file, which consists of
// a main() function which calls RUN_ALL_TESTS() for us.
//
// This runs all the tests you've defined, prints the result, and
// returns 0 if successful, or 1 otherwise.
//
// Did you notice that we didn't register the tests?  The
// RUN_ALL_TESTS() macro magically knows about all the tests we
// defined.  Isn't this convenient?