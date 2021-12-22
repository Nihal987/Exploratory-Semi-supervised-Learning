class SelfTrain():
    def __init__(self,base_model) -> None:
        self.model = base_model

    def fit(self,X_train,y_train,X_ulab,iterations=10,threshold=0.75):
        
        if X_ulab is None:
            self.model.fit(X_train,y_train)
            return
        
        i = 0
        while(len(X_ulab)>0 and iterations > i):
    #         print(f"Ran {i+1} times")
            i += 1
            self.model.fit(X_train,y_train)
            
            y_pred_prob = self.model.predict_proba(X_ulab)
            y_pred = self.model.predict(X_ulab)
            
            conf_index = np.where(np.amax(y_pred_prob,1)>threshold)[0]
            
            if len(conf_index) == 0:
                print("Confidence Interval too high")
                break
            
            X_train = np.append(X_train,X_ulab[conf_index],axis=0)
            y_train = np.append(y_train,y_pred[conf_index])
            
            inv_conf_index = np.where(np.amax(y_pred_prob,1)<=threshold)
            
            X_ulab = X_ulab[inv_conf_index] #removing the pseudo labelled data 
            
            iterations -= 1

    def predict(self,X):
        return self.model.predict(X)