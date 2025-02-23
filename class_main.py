from observation_parser import ParsingObservations
from GridDisplacementModel import GridDisplacementModel
from GridModelEval import ModelEvaluation


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
    print(dead_grid_model.mu)
    print(dead_grid_model.cov_matrix)
    train_dead_with_dead_probs, min_dead_probs=dead_grid_model.compute_probabilities(train_dead_observation)
    train_dead_with_dead=dead_grid_model.add_labels_to_dict(train_dead_with_dead_probs, DEAD)
    
    
    alive_file_list=['AliveObjectXYs1a.txt']
    alive_file_loader = ParsingObservations(alive_file_list)
    alive_observations=alive_file_loader.observations
    
    train_alive_observation,test_alive_observations=alive_file_loader.prepare_train_test(alive_observations,train_ratio=0.8)
    print(len(train_alive_observation))
    train_alive_with_dead_probs, min_alive_probs=dead_grid_model.compute_probabilities(train_alive_observation)
    print(len(train_alive_with_dead_probs))
    train_alive_with_dead=dead_grid_model.add_labels_to_dict(train_alive_with_dead_probs, ALIVE)
    
    
    evalaute_model= ModelEvaluation(train_dead_with_dead)

    evalaute_model.combine_data(train_alive_with_dead)
    evalaute_model.run_thresholding()
    evalaute_model.predict_probabilities_dictionary_update()
    
    evalaute_model.find_the_misclassified_obj()
    
    #run this to see if normalization working or not
    '''
    dead_grid_model_norm = GridDisplacementModel(grid_rows=5, grid_cols=5, max_x=4128, max_y=2196,use_normalization=True)
    dead_grid_model_norm.covariance(train_dead_observation)
    print(dead_grid_model_norm.mu)
    #print(dead_grid_model.cov_matrix)
    train_dead_with_dead_probs, min_dead_probs=dead_grid_model_norm.compute_probabilities(train_dead_observation)
    #print(train_dead_with_dead_probs)
    print(len(train_dead_with_dead_probs))
    train_dead_with_dead_norm=dead_grid_model_norm.add_labels_to_dict(train_dead_with_dead_probs, DEAD)
    #print(train_dead_with_dead_norm)
    '''
    #this will result combined mu, covariance
    '''
    organic_file_loader = ParsingObservations([dead_file_list[1]])
    organic_observations=organic_file_loader.observations
   
    train_organic_observation,test_organic_observations=organic_file_loader.prepare_train_test(organic_observations,train_ratio=0.8)
    
    organic_grid_model = GridDisplacementModel()
    organic_grid_model.covariance(train_organic_observation)
    print(organic_grid_model.mu)
    print(organic_grid_model.cov_matrix)
    
    combine_dead_organic_model=GridDisplacementModel()
    combined_models=combine_dead_organic_model.add_models(dead_grid_model,organic_grid_model)
    print(combined_models.mu)
    print(combined_models.cov_matrix)
    '''
    #this throwing errors because not normalization; work in progress
    '''
    train_dead_with_dead_probs, min_dead_probs=combined_models.compute_probabilities(train_dead_observation)
    print(len(train_dead_observation))
    print(len(train_dead_with_dead_probs))
   
    
    train_organic_with_dead_probs, organic_min_dead_probs=combined_models.compute_probabilities(train_organic_observation)
    print(len(train_organic_observation))
    print(len(train_organic_with_dead_probs))
    train_organic_with_dead=dead_grid_model.add_labels_to_dict(train_organic_with_dead_probs, DEAD)
    '''
    #print(train_dead_with_dead)
   
    
    
    
    