import numpy 
import scipy.stats 
import math 

#String literals to constants
TRUE_LABELS = "true_labels"
LOG_PDFS='log_pdfs'
DEAD='d'
ALIVE='a'

class GridDisplacementModel:
    def __init__(self, grid_rows=5, grid_cols=5, max_x=4128, max_y=2196,use_normalization=False):
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
        
        # "normalized" displacement statistics
        self.dx_sum = 0
        self.dy_sum = 0
        
        self.dx_squared_sum=0
        self.dy_squared_sum=0
        
        self.total_n = 0
        #print(f"seeing the initialized values: {self.cov_mat}")
        
        #stats for combining further stats:
        self.norm_dx_sum = [[ 0 for _ in range(grid_cols)] for _ in range(grid_rows)]
        self.norm_dy_sum = [[ 0 for _ in range(grid_cols)] for _ in range(grid_rows)]
        
        self.norm_dx_squared_sum=[[ 0 for _ in range(grid_cols)] for _ in range(grid_rows)]
        self.norm_dy_squared_sum=[[ 0 for _ in range(grid_cols)] for _ in range(grid_rows)]
        #to compute the cross_product of dx*dy to calculate the covariance matrix
        self.sum_norm_dxdy = [[ 0 for _ in range(grid_cols)] for _ in range(grid_rows)]
    def add_models(self, *others):

        for o in others:
            assert self.grid_rows == o.grid_rows
            assert self.grid_cols == o.grid_cols
            assert self.max_x == o.max_x
            assert self.max_y == o.max_y

        combined = GridDisplacementModel(self.grid_rows, self.grid_cols, self.max_x, self.max_y)
                
        # TODO: walk through all grid cells and combine the sufficient
        # statistics of "self" and "other" and put the result in "combined"
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # Sum of observations for each grid cell
                cell_n = self.n[row][col] + sum(o.n[row][col] for o in others)  # Compute total observations
                cell_sum_dx=0
                cell_sum_dy=0
                
                cell_sum_square_dx=0
                cell_sum_square_dy=0
                
                cell_sum_dxdy=0
                
                cell_product_dx=1
                cell_product_dy=1
                
                if cell_n > 0:  # Avoid division by zero
                    weighted_mu_x = self.n[row][col] * self.mu[row][col][0]  
                    weighted_mu_y = self.n[row][col] * self.mu[row][col][1]  

                    for o in others:  # Iterate over all other models
                        weighted_mu_x += o.n[row][col] * o.mu[row][col][0] 
                        weighted_mu_y += o.n[row][col] * o.mu[row][col][1]  

                    # Compute final weighted mean
                    combined.mu[row][col] = (weighted_mu_x / cell_n,  weighted_mu_y / cell_n)

                    # Compute covariance matrix
                    combined_cov = numpy.zeros((2, 2))
                    for o in others:  # Iterate over all other models
                        cell_sum_dx += o.norm_dx_sum[row][col]
                        cell_sum_dy += o.norm_dy_sum[row][col]
                        
                        cell_sum_square_dx += o.norm_dx_squared_sum[row][col]
                        cell_sum_square_dy+= o.norm_dy_squared_sum[row][col]
                        
                        cell_sum_dxdy+=o.sum_norm_dxdy[row][col]
                        
                        cell_product_dx *= o.norm_dx_sum[row][col]
                        cell_product_dy *= o.norm_dy_sum[row][col]
                        
                    cell_var_x=cell_sum_square_dx-(2*combined.mu[row][col][0]*cell_product_dx)+(cell_n*(combined.mu[row][col][0]**2))
                    cell_var_y=cell_sum_square_dy-(2*combined.mu[row][col][1]*cell_product_dy)+(cell_n*(combined.mu[row][col][1]**2))
                    cell_cov_xy=cell_sum_dxdy-(combined.mu[row][col][0]*cell_sum_dy)-(combined.mu[row][col][1]*cell_sum_dx)+(cell_n*(combined.mu[row][col][0]*combined.mu[row][col][1]))
                    
                    combined_cov=numpy.array([[cell_var_x, cell_cov_xy], [cell_cov_xy, cell_var_y]])
                    combined.cov_matrix[row][col] = combined_cov 

                    # Assign total count
                    combined.n[row][col] = cell_n
        return combined

    def calculate_displacements(self, observations):
        #to keep the displacements in the grid formats
        grid_dis = [[[] for _ in range(self.num_rows())] for _ in range(self.num_cols())]
        
        for obj_id, obs in observations.items():
            for i in range(len(obs) - 1):
                dframe = obs[i+1][0] - obs[i][0]
                #to do: dframe<=0 continue logging error
                if dframe>0:
                
                    dx = obs[i+1][1] - obs[i][1]
                    dy = obs[i+1][2] - obs[i][2]

                    grid_row, grid_cell = self.find_grid_cell(obs[i][1],
                                                      obs[i][2])
                    grid_pos=grid_dis[grid_row][grid_cell]
                    
                    self.n[grid_row][grid_cell] += 1
                    
                    dx=dx/dframe
                    dy=dy/dframe
                    grid_pos.append((dx,dy))
                    
                    #calculating for normalization:
                    self.dx_sum += dx
                    self.dy_sum += dy
                
                    self.dx_squared_sum+= (dx **2)
                    self.dy_squared_sum+= (dy **2)
                    
                
                    self.total_n += 1
                    
                    if self.use_normalization==False:
                        self.norm_dx_sum[grid_row][grid_cell]+=dx
                        self.norm_dy_sum[grid_row][grid_cell]+=dy
                        self.norm_dx_squared_sum[grid_row][grid_cell]+=(dx**2)
                        self.norm_dy_squared_sum[grid_row][grid_cell]+=(dy**2)
                        self.sum_norm_dxdy[grid_row][grid_cell]+=(dx*dy)
                else:
                    print(f"distance of frame is getting invalid values for calculation: {dframe}")
        
        if self.use_normalization:
        
            grid_dis=self.apply_normalization(grid_dis)
        
        return grid_dis
        
    def normalization(self):
        dx_norm = dy_norm = sx_norm = sy_norm = 0

        if self.use_normalization:
            dx_norm = self.dx_sum / self.total_n
            dy_norm = self.dy_sum / self.total_n
            # FIXME: standard deviation normalizations
            var_x = (self.dx_squared_sum / self.total_n) - (dx_norm **2)
            var_y = (self.dy_squared_sum / self.total_n) - (dy_norm **2)
            sx_norm = numpy.sqrt(var_x) if var_x > 0 else 1 #avoiding 0
            sy_norm = numpy.sqrt(var_y) if var_y > 0 else 1

        return dx_norm, dy_norm, sx_norm, sy_norm
    
    def apply_normalization(self,grid_displacements):
    
        dx_norm, dy_norm, sx_norm, sy_norm=self.normalization()     
       
        for row in range(self.num_rows()):
            for col in range(self.num_cols()):
                if len(grid_displacements[row][col]) > 0:  # enough values to normalize
                    normalized_displacements = []
                    #print(f"before normalizations: sizes are: {len(grid_displacements[row][col])}")
                    for dx, dy in grid_displacements[row][col]:
                        norm_dx = (dx - dx_norm) / sx_norm 
                        norm_dy = (dy - dy_norm) / sy_norm
                        normalized_displacements.append((norm_dx, norm_dy))
                        #needed for combining the stats for later:
                        self.norm_dx_sum[row][col]+=norm_dx
                        self.norm_dy_sum[row][col]+=norm_dy
                        self.norm_dx_squared_sum[row][col]+=(norm_dx**2)
                        self.norm_dy_squared_sum[row][col]+=(norm_dy**2)
                        self.sum_norm_dxdy[row][col]+=(norm_dx*norm_dy)
                        
                    grid_displacements[row][col] = normalized_displacements  
                #print(f"after normalizations: sizes are: {len(grid_displacements[row][col])}")
        return grid_displacements

        
    def covariance(self,observations):
       
        grid_displacements=self.calculate_displacements(observations)
        
        for row in range(self.num_rows()):
            for col in range(self.num_cols()):
                n=self.n[row][col]
                #print(row,col)
                if n>1:
                    if n<30:
                        print(f"at grid {row}{col} obs are: {n} less than 30")
                        assert n == len(grid_displacements[row][col]), f"Mismatch: {n} is but items are: {len(grid_displacements[row][col])}"        
                            
                    dxdy_items = numpy.array(grid_displacements[row][col])
                        
                    cell_mu = numpy.mean(dxdy_items, axis=0)
                    self.mu[row][col]=cell_mu
                    #print(cell_mu,cell_mu.shape)
                
                    cell_cov_matrix = numpy.cov(dxdy_items.T)
                    self.cov_matrix[row][col]=cell_cov_matrix
                                
                    #print(cell_cov_matrix,cell_cov_matrix.shape)
                                
                else:
                    print(f"at grid {row}{col} obs are: {n} not enough to calculate")
                
        return 
    
    
    def probability(self, x, y, dx, dy):
        grid_row, grid_col = self.find_grid_cell(x, y)
        cell_mu = self.mu[grid_row][grid_col]
        cell_cov_matrix = self.cov_matrix[grid_row][grid_col]
        n = self.n[grid_row][grid_col]
        
        if n>=1:
            #print(f"current cell [{grid_row}][{grid_col}] mu is: {cell_mu} sigma is: {cell_cov_matrix} {dx,dy}")
            # TODO: create a 2-dimensional Gaussian distribution and use it to calculate a probability for (dx, dy)
            mvn = scipy.stats.multivariate_normal(mean=cell_mu , cov=cell_cov_matrix) #to do use the library name
            probability=mvn.pdf((dx,dy))
            
            #print(f"current cell [{grid_row}][{grid_col}] mu is: {cell_mu} sigma is: {cell_cov_matrix}  probabilities: {probability} for {dx,dy}")
        return probability
    
    def compute_probabilities(self, observations):
        if self.use_normalization==True:
            grid_displacements=self.calculate_displacements(observations)
            dx_norm, dy_norm, sx_norm, sy_norm=self.normalization()
            
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
            #print(obj_id,len(obj_probabilities),len(obs))
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

