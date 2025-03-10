from observation_parser import ParsingObservations
from GridDisplacementModel import GridDisplacementModel
from GridModelEval import ModelEvaluation
from GridProbabilityCalculator import GridProbabilityCalculator

TRUE_LABELS = "true_labels"
LOG_PDFS='log_pdfs'
DEAD='d'
ALIVE='a'

if __name__ == "__main__":
  
    dead_file_list = ['DeadObjectXYs.txt','1-6-25_1_ObjectXYs.txt']
    dead_file_loader = ParsingObservations([dead_file_list[0]])
    dead_observations=dead_file_loader.observations
   
    train_dead_observation,test_dead_observations=dead_file_loader.prepare_train_test(dead_observations,train_ratio=0.8)
    
    dead_grid_model = GridDisplacementModel()
    dead_grid_model.covariance(train_dead_observation)
    dead_grid_model.norm_dx,dead_grid_model.norm_dy,dead_grid_model.norm_sx,dead_grid_model.norm_sy=dead_grid_model.normalization()
    #print(f"from main: {dead_grid_model.norm_dx},{dead_grid_model.norm_dy},{dead_grid_model.norm_sx},{dead_grid_model.norm_sy}")
    #print(dead_grid_model.mu)
    #print(dead_grid_model.cov_matrix)
    
    
    alive_file_list=['AliveObjectXYsp1.txt']
    alive_file_loader = ParsingObservations(alive_file_list)
    alive_observations=alive_file_loader.observations
    
    train_alive_observation,test_alive_observations=alive_file_loader.prepare_train_test(alive_observations,train_ratio=0.8)
    
    alive_grid_model = GridDisplacementModel()
    alive_grid_model.covariance(train_alive_observation)
    alive_grid_model.norm_dx,alive_grid_model.norm_dy,alive_grid_model.norm_sx,alive_grid_model.norm_sy=alive_grid_model.normalization()
    #print(len(train_alive_observation))
    
  
    
    organic_file_loader = ParsingObservations([dead_file_list[1]])
    organic_observations=organic_file_loader.observations
   
    train_organic_observation,test_organic_observations=organic_file_loader.prepare_train_test(organic_observations,train_ratio=0.8)
    
    organic_grid_model = GridDisplacementModel()
    organic_grid_model.covariance(train_organic_observation)
    organic_grid_model.norm_dx,organic_grid_model.norm_dy,organic_grid_model.norm_sx,organic_grid_model.norm_sy=organic_grid_model.normalization()
    #print(organic_grid_model.mu)
    #print(organic_grid_model.cov_matrix)
    
    #this will result combined mu, covariance
    combine_dead_organic_model=GridDisplacementModel()
    combined_models=combine_dead_organic_model.add_models(dead_grid_model,organic_grid_model)
    print("Combined Models are: ")
    #print(combined_models.mu)
    #print(combined_models.cov_matrix)
    
    outlier_probability_calculator=GridProbabilityCalculator()
    outlier_probability_calculator.n=combined_models.n
    outlier_probability_calculator.mu=combined_models.mu
    outlier_probability_calculator.cov_matrix=combined_models.cov_matrix
    #print(outlier_probability_calculator.mu)
    #print(outlier_probability_calculator.cov_matrix)
    
    
    train_dead_with_dead_probs, min_dead_probs=outlier_probability_calculator.compute_probabilities(train_dead_observation,dead_grid_model.norm_dx,dead_grid_model.norm_dy,dead_grid_model.norm_sx,dead_grid_model.norm_sy)
    train_dead_with_dead=outlier_probability_calculator.add_labels_to_dict(train_dead_with_dead_probs, DEAD)
    #print(len(train_dead_with_dead),len(train_dead_observation))
    train_organic_with_dead_probs, min_organic_probs=outlier_probability_calculator.compute_probabilities(train_organic_observation,organic_grid_model.norm_dx,organic_grid_model.norm_dy,organic_grid_model.norm_sx,organic_grid_model.norm_sy)
    train_organic_with_dead=outlier_probability_calculator.add_labels_to_dict(train_organic_with_dead_probs, DEAD)
    #print(len(train_organic_observation),len(train_organic_with_dead))
    train_alive_with_dead_probs, min_alive_probs=outlier_probability_calculator.compute_probabilities(train_alive_observation,alive_grid_model.norm_dx,alive_grid_model.norm_dy,alive_grid_model.norm_sx,alive_grid_model.norm_sy)
    train_alive_with_dead=outlier_probability_calculator.add_labels_to_dict(train_alive_with_dead_probs, ALIVE)
    print(len(train_alive_observation),len(train_alive_with_dead))
    
    
    
    evalaute_model= ModelEvaluation(train_dead_with_dead)

    evalaute_model.combine_data(train_organic_with_dead)
    evalaute_model.combine_data(train_alive_with_dead)
    evalaute_model.run_thresholding()
    evalaute_model.predict_probabilities_dictionary_update()
    
    evalaute_model.find_the_misclassified_obj()
    
   
    
    
    
    