#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iomanip>

struct vector_3d {
	double x;
	double y;
	double z;
};

typedef struct linear_regression_results {
	double offset_x;
	double offset_y;
	double offset_z;
	double estimated_radius;
} SPHERE_LINEAR_REGRESSION_RESULTS;
	
double find_value(const std::string& line, const std::string& key) {
	size_t pos = line.find(key);
	if (pos != std::string::npos) {
		size_t value_start = pos + key.length(); 
		try {
			return std::stod(line.substr(value_start));
		} catch (const std::invalid_argument& e) {
			std::cerr << "Could not convert value for key '" << key << "' in line: " << line << std::endl;
		}
	}
	return 0.0;
}

std::vector<vector_3d> read_data(const char* data_file_path) {
	std::vector<vector_3d> points;
	std::ifstream file(data_file_path);
	std::string line;

	if (!file.is_open()) {
		std::cerr << "Error: Could not open file " << data_file_path << std::endl;
		return points;
	}

	while (std::getline(file, line)) {
		if (line.find("type=DATA_MAG") != std::string::npos) {
		vector_3d point;
		point.x = find_value(line, "x=");
		point.y = find_value(line, "y=");
		point.z = find_value(line, "z=");
		points.push_back(std::move(point));
		}
	}

	file.close();
	return points;
}

vector_3d mean_middle_point(const std::vector<vector_3d>& vec_array) {
	double x_cord_sum = 0;
	double y_cord_sum = 0;
	double z_cord_sum = 0;

	for (size_t i = 0; i < vec_array.size(); i++) {
		x_cord_sum += vec_array.at(i).x;
		y_cord_sum += vec_array.at(i).y;
		z_cord_sum += vec_array.at(i).z;
	}

	double mean_x_cord = x_cord_sum / vec_array.size();
	double mean_y_cord = y_cord_sum / vec_array.size();
	double mean_z_cord = z_cord_sum / vec_array.size();

	return vector_3d{mean_x_cord, mean_y_cord, mean_z_cord};

}

void centralize_points(std::vector<vector_3d>& vec_array, const vector_3d& vec_mean) {
	for (size_t i = 0; i < vec_array.size(); i++) {
		vec_array.at(i).x -= vec_mean.x;
		vec_array.at(i).y -= vec_mean.y;
		vec_array.at(i).z -= vec_mean.z;
	}
}


std::vector<std::vector<double>> invert_matrix(std::vector<std::vector<double>> matrix) {
	int n = matrix.size();
	if (n == 0 || matrix[0].size() != n) {
		throw std::runtime_error("Matrix must be square.");
	}

	// Creating extended matrix [matrix | I]
	// I matrix stands for identity matrix
	std::vector<std::vector<double>> augmented(n, std::vector<double>(2 * n));
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
		augmented[i][j] = matrix[i][j];
		}
		augmented[i][i + n] = 1;
	}

	for (int i = 0; i < n; ++i) {
		int pivot_row = i;

		for (int k = i + 1; k < n; ++k) {
			if (std::abs(augmented[k][i]) > std::abs(augmented[pivot_row][i])) {
				pivot_row = k;
			}
		}
		std::swap(augmented[i], augmented[pivot_row]);

		if (std::abs(augmented[i][i]) < 1e-9) {
			throw std::runtime_error("Matrix is singular and cannot be inverted.");
		}

		double pivot = augmented[i][i];
		for (int j = i; j < 2 * n; ++j) {
			augmented[i][j] /= pivot;
		}

		for (int k = 0; k < n; ++k) {
			if (k != i) {
				double factor = augmented[k][i];
				for (int j = i; j < 2 * n; ++j) {
					augmented[k][j] -= factor * augmented[i][j];
				}
			}
		}
	}

	std::vector<std::vector<double>> inverse(n, std::vector<double>(n));
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			inverse[i][j] = augmented[i][j + n];
		}
	}
	return inverse;
}

std::vector<double> multiply_matrix_vector(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vec) {
	int rows = matrix.size();
	int cols = matrix[0].size();

	std::vector<double> result(rows, 0.0);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
		result[i] += matrix[i][j] * vec[j];
		}
	}
	return result;
}


SPHERE_LINEAR_REGRESSION_RESULTS SPHERE_linear_regression(std::vector<vector_3d>& vec_array) {
	vector_3d mean_point = mean_middle_point(vec_array);
	centralize_points(vec_array, mean_point);
	
	/*
	* Summing x,y,z coordinate of each point in data set
	*/

	// N constant is used in M matrix 
	// N is just number of points in our data set
	double N = static_cast<double>(vec_array.size());

	double sum_x = 0;
	double sum_y = 0;
	double sum_z = 0;

	double sum_x2 = 0;
	double sum_y2 = 0;
	double sum_z2 = 0;

	double sum_xy = 0;
	double sum_xz = 0;
	double sum_yz = 0;

	double sum_xf = 0;
	double sum_yf = 0;
	double sum_zf = 0;
	double sum_f = 0;

	for (size_t i = 0; i < vec_array.size(); i++) {
		double x_val = vec_array.at(i).x;
		double y_val = vec_array.at(i).y;
		double z_val = vec_array.at(i).z;

		// f_value stands for x^2 + y^2 + z^2
		// Above formula can be derived from sphere equation
		double f_value = x_val * x_val + y_val * y_val + z_val * z_val;

		sum_x += x_val;
		sum_y += y_val;
		sum_z += z_val;
		
		sum_x2 += (x_val * x_val);
		sum_y2 += (y_val * y_val);	
		sum_z2 += (z_val * z_val);

		sum_xy += (x_val * y_val);
		sum_xz += (x_val * z_val);
		sum_yz += (y_val * z_val);

		sum_xf += x_val * f_value;
		sum_yf += y_val * f_value;
		sum_zf += z_val * f_value;
		sum_f += f_value;
	}

	// Constructing M Matrix from above sums
	std::vector<std::vector<double>> MATRIX_M {
		{2 * sum_x2, 2 * sum_xy, 2 * sum_xz, sum_x},
        	{2 * sum_xy, 2 * sum_y2, 2 * sum_yz, sum_y},
        	{2 * sum_xz, 2 * sum_yz, 2 * sum_z2, sum_z},
        	{sum_x, sum_y, sum_z, N}
	};

	// Constructing R Matrix from above sums
	std::vector<double> VECTOR_R {
		sum_xf, sum_yf, sum_zf, sum_f
	};

	std::vector<std::vector<double>> M_inv = invert_matrix(MATRIX_M);
	std::vector<double> V = multiply_matrix_vector(M_inv, VECTOR_R);

	double xc_centered = V[0];
	double yc_centered = V[1];
	double zc_centered = V[2];
	double k = V[3];

	SPHERE_LINEAR_REGRESSION_RESULTS results;

	results.offset_x = xc_centered + mean_point.x;
	results.offset_y = yc_centered + mean_point.y;
	results.offset_z = zc_centered + mean_point.z;
	results.estimated_radius = std::sqrt(k + xc_centered * xc_centered + yc_centered * yc_centered + zc_centered * zc_centered);
	
	return results;
}

int main() {
	const char* file_path = "data.txt";
	std::vector<vector_3d> data_points = read_data(file_path);

	SPHERE_LINEAR_REGRESSION_RESULTS results = SPHERE_linear_regression(data_points);

	std::cout << std::fixed << std::setprecision(15);
	std::cout << "Offset X: " << results.offset_x << std::endl;
	std::cout << "Offset Y: " << results.offset_y << std::endl;
	std::cout << "Offset Z: " << results.offset_z << std::endl;
	std::cout << "Radius: " << results.estimated_radius << std::endl;

	return 0;
}