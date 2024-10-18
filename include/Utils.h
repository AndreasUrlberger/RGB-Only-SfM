#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <iostream>

std::tuple<double, double> calculateReprojectionError(std::vector<cv::Point2f>& observed, std::vector<cv::Point2f>& predicted)
{
    std::vector<double> errors;
    for (size_t i = 0; i < observed.size(); ++i)
    {
        errors.push_back(cv::norm(observed[i] - predicted[i]));
    }

    double mean_error = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
    std::vector<double> squares(errors.size());
    std::transform(errors.begin(), errors.end(), squares.begin(), [](double x) {
        return x * x;
    });
    double rms_error = std::sqrt(std::accumulate(squares.begin(), squares.end(), 0.0)/squares.size());
    return std::make_tuple(mean_error, rms_error);
}

#endif // UTILS_H