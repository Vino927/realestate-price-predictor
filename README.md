# re-price-predictor

Real Estate Price Prediction Project
This Real Estate Price Prediction project is designed to predict the prices of real estate properties based on various features such as the number of bedrooms, bathrooms, lot size, zip code, and house size. The project is structured into modular components for ease of understanding and maintenance.

## Project Structure
The project is divided into the following modules:

dataloader.py: Handles data loading, cleaning, and preprocessing.
featurescaling.py: Scales the features and target variable for neural network input.
predictor.py: Builds and trains the neural network model for price prediction.
evaluator.py: Evaluates the model's performance and visualizes the prediction results.

# Setup Instructions
To run this project, you will need Python 3.6 or later. It is recommended to use a virtual environment for installing the dependencies.

### Clone the repository to your local machine:
```
bash
```
### Copy code
```
[git clone https://your-repository-url.git](https://github.com/Vino927/re-price-predictor.git)
```

### Install the required dependencies:
```
pip install -r requirements.txt
```
## Usage
To use the project, you must follow these steps sequentially:

Data Preprocessing: Load and preprocess your dataset using DataPreprocessor.
Feature Scaling: Scale your features and target variable with FeatureScaler.
Model Training: Train the real estate price prediction model using RealEstatePricePredictor.
Model Evaluation: Evaluate the model's performance and visualize the results using ModelEvaluator.


## License
This project is licensed under the GNU General Public License (GPL) v3.0. The GPL is a free, copyleft license that allows software to be freely used, modified, and shared under the same terms. GPL v3.0 further strengthens this approach by ensuring that all derived works are also distributed under the GPL, protecting the software's freedom and the rights of users of GPL-licensed code.

For more details, see the full GPL license text here: [GPL v3.0 License](https://www.gnu.org/licenses/gpl-3.0.html).

