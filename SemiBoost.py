class SemiBoostClassifier():

    def __init__(self, base_model =SVC()):

        self.BaseModel = base_model
        self.weights = None

    def fit(self, X, y,
            n_neighbors=4, n_jobs = 1,
            max_models = 15,
            sample_percent = 0.5,
            sigma_percentile = 90,
            similarity_kernel = 'rbf',
            verbose = False):
        
        #Localize labeled data
        label_index = np.argwhere((y == 0) | (y == 1)).flatten()
        unlabel_index = np.array([i for i in np.arange(len(y)) if i not in label_index])

        # For Fully Supervised Learning
        if unlabel_index.size == 0:
            index = np.where(y!=-1)
            X,y = X[index],y[index]
            self.BaseModel.fit(X,y)
            return 

        # First we need to create the similarity matrix
        if similarity_kernel == 'knn':

            graph = neighbors.kneighbors_graph(X,
                                                n_neighbors=n_neighbors,
                                                mode='distance',
                                                include_self=True,
                                                n_jobs=n_jobs)

            graph = sparse.csr_matrix(graph)

        elif similarity_kernel == 'rbf':
            # First aprox
            graph = np.sqrt(rbf_kernel(X, gamma = 1))
            # set gamma parameter as the 15th percentile
            sigma = np.percentile(np.log(graph), sigma_percentile)
            sigma_2 = (1/sigma**2)*np.ones((graph.shape[0],graph.shape[0]))
            graph = np.power(graph, sigma_2)
            # Matrix to sparse
            graph = sparse.csr_matrix(graph)

        else:
            print('No kernel type ', similarity_kernel)

        # Initialise variables
        self.models = []
        self.weights = []
        H = np.zeros(unlabel_index.shape[0])

        # Loop for adding sequential models
        for t in range(max_models):
            # Calculate p_i and q_i for every sample
            p_1 = np.einsum('ij,j', graph[:,label_index].todense(), (y[label_index]==1))[unlabel_index]*np.exp(-2*H)
            p_2 = np.einsum('ij,j', graph[:,unlabel_index].todense(), np.exp(H))[unlabel_index]*np.exp(-H)
            p = np.add(p_1, p_2)
            p = np.squeeze(np.asarray(p))

            q_1 = np.einsum('ij,j', graph[:,label_index].todense(), (y[label_index]==-1))[unlabel_index]*np.exp(2*H)
            q_2 = np.einsum('ij,j', graph[:,unlabel_index].todense(), np.exp(-H))[unlabel_index]*np.exp(H)
            q = np.add(q_1, q_2)
            q = np.squeeze(np.asarray(q))

            # Compute predicted label z_i
            z = np.sign(p-q)
            z_conf = np.abs(p-q)
            # Sample sample_percent most confident predictions
            # Sampling weights
            sample_weights = z_conf/np.sum(z_conf)
            # If there are non-zero weights
            if np.any(sample_weights != 0):
                pick = np.random.choice(np.arange(z.size),
                                              size = int(sample_percent*unlabel_index.size),
                                              p = sample_weights,
                                              replace = False)
                sample_index = unlabel_index[pick]

            else:
                print('No similar unlabeled observations left.')
                break

            # Create new X_t, y_t
            total_sample = np.concatenate([label_index,sample_index])
            X_t = X[total_sample,]
            np.put(y, sample_index, z[pick])# Include predicted to train new model
            y_t = y[total_sample]

            # Fit BaseModel to samples using predicted labels
            # Fit model to unlabeled observations
            clf = self.BaseModel
            clf.fit(X_t, y_t)
            # Make predictions for unlabeled observations
            h = clf.predict(X[unlabel_index])

            # Refresh indexes
            label_index = total_sample
            unlabel_index = np.array([i for i in np.arange(len(y)) if i not in label_index])

            if verbose:
                print('There are still ', unlabel_index.shape[0], ' unlabeled observations')

            # Compute weight (a) for the BaseModel as in (12)
            e = (np.dot(p,h==-1) + np.dot(q,h==1))/(np.sum(np.add(p,q)))
            a = 0.25*np.log((1-e)/e)
            # Update final model
            # If a<0 the model is not converging
            if a<0:
                if verbose:
                    print('Problematic convergence of the model. a<0')
                break

            # Save model
            self.models.append(clf)
            #save weights
            self.weights.append(a)
            # Update
            H = np.zeros(len(unlabel_index))
            w = np.sum(self.weights)
            for i in range(len(self.models)):
                H = np.add(H, self.weights[i]*self.models[i].predict(X[unlabel_index]))

            # Breaking conditions

            # Maximum number of models reached
            if (t==max_models) & verbose:
                print('Maximum number of models reached')

            # If no samples are left without label, break
            if unlabel_index.size <= 1:
                if verbose:
                    print('All observations have been labeled')
                    print('Number of iterations: ',t + 1)
                break

        if verbose:
            print('\n The model weights are \n')
            print(self.weights)



    def predict(self, X):
        if self.weights is None:
            return self.BaseModel.predict(X)
        
        estimate = np.zeros(X.shape[0])
        # Predict weighting each model
        w = np.sum(self.weights)
        for i in range(len(self.models)):
            estimate = np.add(estimate, self.weights[i]*self.models[i].predict(X))
        estimate = np.array(list(map(lambda x: 1 if x>0 else 0, estimate)))
        estimate = estimate.astype(int)
        return estimate
