import numpy 
import scipy.stats 
import math 

#String literals to constants
TRUE_LABELS = "true_labels"
LOG_PDFS='log_pdfs'
DEAD='d'
ALIVE='a'

class GridProbabilityCalculator:
    def __init__(self, grid_rows=5, grid_cols=5, max_x=4128, max_y=2196,use_normalization=True):
        # self.n represents the number of observations for each cell
        self.n = [[ 0 for _ in range(grid_cols)] for _ in range(grid_rows)]

        # self.mu represents mu_x,mu_y for each cell
        self.mu = [[(0, 0) for _ in range(grid_cols)] for _ in range(grid_rows)]

        # self.cov_mat represents the covariance for the each cell
        #to do 2*2 matrix initiliaziation
        self.cov_matrix = [[numpy.zeros((2, 2)) for _ in range(grid_cols)] for _ in range(grid_rows)]
        
        self.max_x = max_x
        self.max_y = max_y
        
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        # TODO: add statistics for normalizing standard deviation

        self.use_normalization = use_normalization
        
    
  
    
    def compute_probabilities(self, observations,dx_norm, dy_norm, sx_norm, sy_norm):
            
        probabilities={}
        minimum_probs=[]
        empty_obs=0
        
        for obj_id, obs in observations.items():
            obj_probabilities=[]
            for i in range(len(obs) - 1):
                x,y=obs[i][1],obs[i][2]
                dframe = obs[i+1][0] - obs[i][0]
                dx = obs[i+1][1] - obs[i][1]
                dy = obs[i+1][2] - obs[i][2]
                if dframe>0:
                    dx,dy=(dx/dframe),(dy/dframe)
                    if self.use_normalization==True:
                        norm_dx = (dx - dx_norm) / sx_norm 
                        norm_dy = (dy - dy_norm) / sy_norm
                        probs=self.probability(x, y, norm_dx, norm_dy)
                        
                        obj_probabilities.append(probs)
                    else:
                        probs=self.probability(x, y, dx, dy)
                        obj_probabilities.append(probs)
                        
            assert len(obj_probabilities) == len(obs)-1, f"Mismatch: {obj_id} has {len(obj_probabilities)} probabilities but {len(obs)-1} observations!"
            if len(obs)-1==0:
                empty_obs+=1
            
            log_obj_probabilities=self.log_probability(obj_probabilities)
            #print(obj_id,len(log_obj_probabilities),len(obj_probabilities),len(obs))
            
            if len(log_obj_probabilities)>=1:
                probabilities[obj_id]={LOG_PDFS:log_obj_probabilities}
                minimum_probs.append(min(log_obj_probabilities))
                
        print(f"emptys are: {empty_obs}")  
        
        return probabilities,minimum_probs    
    def log_probability(self, curr_pdf_list):
        #print(f"before math:{len(curr_pdf_list)}, {curr_pdf_list}")   
        log_values = [math.log(x) for x in curr_pdf_list if x != 0] 
        #print(f"after math: {len(curr_pdf_list)}, {len(log_values)}")
        #log_sum_values = np.sum(log_values)
        return log_values
        
    def probability(self, x, y, dx, dy):
    
        grid_row, grid_col = self.find_grid_cell(x, y)
        cell_mu = self.mu[grid_row][grid_col]
        cell_cov_matrix = self.cov_matrix[grid_row][grid_col]
        n = self.n[grid_row][grid_col]
        
        if n>=1:
            
            # TODO: create a 2-dimensional Gaussian distribution and use it to calculate a probability for (dx, dy)
            mvn = scipy.stats.multivariate_normal(mean=cell_mu , cov=cell_cov_matrix) #to do use the library name
            curr_probability=mvn.pdf((dx,dy))
            
            #print(f"current cell [{grid_row}][{grid_col}] mu is: {cell_mu} covariance is: {cell_cov_matrix}  probabilities: {curr_probability} for {dx,dy}")
            return curr_probability
        else:
            print(f"current cell [{grid_row}][{grid_col}] mu is: {cell_mu} sigma is: {cell_cov_matrix}  probabilities: empty for {dx,dy}")
            return 
    def add_labels_to_dict(self, curr_log_pdf_dict, label):
        
        for obj_id in curr_log_pdf_dict:
            curr_log_pdf_dict[obj_id][TRUE_LABELS] =label
        
        return curr_log_pdf_dict
    
    def find_grid_cell(self, x, y):
        grid_row = y * self.num_rows() // self.max_y
        grid_col = x * self.num_cols() // self.max_x
        return grid_row, grid_col

    def num_rows(self):
        return len(self.n)

    def num_cols(self):
        return len(self.n[0])